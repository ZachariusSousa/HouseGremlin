from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_launchers_are_the_only_documented_entry_points():
    for name in ("setup.cmd", "run.cmd", "stop.cmd"):
        assert (ROOT / name).is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ".\\setup" in readme
    assert ".\\run" in readme
    assert ".\\stop" in readme


def test_firmware_forward_and_reverse_outputs_match_robit_wiring():
    source = (ROOT / "firmware" / "robit_controller" / "motors.cpp").read_text(encoding="utf-8")

    assert "void moveForward() {\n  analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);\n  setDirection(false, true, false, true);" in source
    assert "void moveReverse() {\n  analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);\n  setDirection(true, false, true, false);" in source
