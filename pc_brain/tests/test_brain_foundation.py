import asyncio

import pytest

from app.brain_models import (
    ActionIntent,
    BodyState,
    ConversationState,
    EventSource,
    WorkPriority,
)
from app.coordinator import BrainCoordinator
from app.journal import EventJournal
from app.resource_lease import PriorityResourceLease


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_journal_is_ordered_and_restores_state_and_turns(tmp_path):
    journal = EventJournal(tmp_path / "brain.db")
    coordinator = BrainCoordinator(journal)
    correlation_id = coordinator.new_correlation_id()

    coordinator.record_turn("user", "hello", EventSource.browser, correlation_id)
    coordinator.record_turn("assistant", "hello back", EventSource.text_model, correlation_id)
    coordinator.transition(
        correlation_id,
        EventSource.text_model,
        conversation=ConversationState.listening,
    )

    events = journal.list_events()
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)
    assert journal.recent_turns()[0].text == "hello"
    assert BrainCoordinator(EventJournal(tmp_path / "brain.db")).state.conversation == ConversationState.listening


def test_journaled_transient_fault_is_not_reactivated_on_restart(tmp_path):
    journal_path = tmp_path / "brain.db"
    coordinator = BrainCoordinator(EventJournal(journal_path))
    coordinator.state = coordinator.state.model_copy(
        update={"body": BodyState.fault, "safety": "fault"}
    )
    coordinator.record(
        "state.changed",
        EventSource.system,
        "corr-old-fault",
        {"state": coordinator.state.model_dump(mode="json")},
    )
    coordinator.journal.close()

    restored = BrainCoordinator(EventJournal(journal_path))

    assert restored.active_faults == []
    assert restored.state.safety == "normal"
    assert restored.state.body == BodyState.stationary


def test_compatibility_safety_field_is_derived_from_active_faults(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))

    coordinator.transition("corr-legacy", EventSource.system, safety="fault")

    assert coordinator.active_faults == []
    assert coordinator.state.safety == "normal"


def test_recent_model_context_is_bounded_to_twenty_turns(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    for index in range(25):
        coordinator.record_turn("user", f"turn {index}", EventSource.browser, f"corr-{index}")

    messages = coordinator.recent_messages()
    assert len(messages) == 20
    assert messages[0]["content"] == "turn 5"
    assert messages[-1]["content"] == "turn 24"


@pytest.mark.anyio
async def test_action_trace_keeps_one_correlation_id(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    calls = []

    async def executor(action):
        calls.append(action)
        return {"ok": True}

    intent = ActionIntent(
        action={"movement": {"direction": "left"}},
        origin=EventSource.text_model,
        correlation_id="corr-action",
    )
    result = await coordinator.execute_action(intent, executor)

    assert result == {"ok": True}
    assert calls == [intent.action]
    events = coordinator.journal.list_events(correlation_id="corr-action")
    assert {event.event_type for event in events} >= {
        "action.proposed",
        "action.approved",
        "action.completed",
    }
    assert {event.correlation_id for event in events} == {"corr-action"}


@pytest.mark.anyio
async def test_successful_action_clears_its_previous_actuation_fault(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    intent = ActionIntent(
        action={"movement": {"direction": "left"}},
        origin=EventSource.text_model,
        correlation_id="corr-action-recovery",
    )

    async def failing_executor(action):
        raise RuntimeError("control acknowledgement timeout")

    with pytest.raises(RuntimeError, match="acknowledgement timeout"):
        await coordinator.execute_action(intent, failing_executor)

    assert [fault.source for fault in coordinator.active_faults] == ["actuation"]
    assert coordinator.state.safety == "fault"

    async def successful_executor(action):
        return {"ok": True}

    assert await coordinator.execute_action(intent, successful_executor) == {"ok": True}
    assert coordinator.active_faults == []
    assert coordinator.state.safety == "normal"
    assert coordinator.state.body == BodyState.stationary


@pytest.mark.anyio
async def test_stop_is_an_ordinary_serialized_action(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    normal_started = asyncio.Event()
    release_normal = asyncio.Event()
    completed = []

    async def normal_executor(action):
        normal_started.set()
        await release_normal.wait()
        completed.append("normal")
        return {"ok": True}

    async def stop_executor(action):
        completed.append("stop")
        return {"ok": True}

    normal_task = asyncio.create_task(
        coordinator.execute_action(
            ActionIntent(
                action={"movement": {"direction": "forward"}},
                origin=EventSource.text_model,
                correlation_id="normal",
            ),
            normal_executor,
        )
    )
    await normal_started.wait()
    stop_task = asyncio.create_task(
        coordinator.execute_action(
            ActionIntent(
            action={"movement": {"direction": "stop"}},
            origin=EventSource.manual,
            correlation_id="stop",
            priority=WorkPriority.manual_action,
            ),
            stop_executor,
        )
    )
    await asyncio.sleep(0)
    assert not stop_task.done()
    release_normal.set()
    await asyncio.gather(normal_task, stop_task)
    assert completed == ["normal", "stop"]


@pytest.mark.anyio
async def test_waiting_manual_action_runs_before_waiting_model_action(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()
    completed = []

    async def executor(action):
        name = action["name"]
        if name == "blocker":
            blocker_started.set()
            await release_blocker.wait()
        completed.append(name)
        return {"ok": True}

    def intent(name, priority):
        return ActionIntent(
            action={"name": name},
            origin=EventSource.manual if priority == WorkPriority.manual_action else EventSource.text_model,
            correlation_id=name,
            priority=priority,
        )

    blocker = asyncio.create_task(coordinator.execute_action(intent("blocker", WorkPriority.model_action), executor))
    await blocker_started.wait()
    model = asyncio.create_task(coordinator.execute_action(intent("model", WorkPriority.model_action), executor))
    manual = asyncio.create_task(coordinator.execute_action(intent("manual", WorkPriority.manual_action), executor))
    while len(coordinator._action_lease._waiters) < 2:
        await asyncio.sleep(0)
    release_blocker.set()
    await asyncio.gather(blocker, model, manual)

    assert completed == ["blocker", "manual", "model"]


@pytest.mark.anyio
async def test_foreground_resource_lease_requests_background_cancellation():
    lease = PriorityResourceLease()
    background_entered = asyncio.Event()
    release_background = asyncio.Event()
    cancellation_requested = asyncio.Event()

    async def background():
        async with lease.acquire(WorkPriority.background, cancellation_requested.set):
            background_entered.set()
            await release_background.wait()

    async def foreground():
        async with lease.acquire(WorkPriority.foreground):
            return "foreground"

    background_task = asyncio.create_task(background())
    await background_entered.wait()
    foreground_task = asyncio.create_task(foreground())
    await asyncio.wait_for(cancellation_requested.wait(), 1)
    assert not foreground_task.done()
    release_background.set()
    assert await foreground_task == "foreground"
    await background_task
