import numpy as np

from src.phase_e_teachers import TeacherCandidate, select_teacher_labels


def test_three_service_first_teachers_deduplicate_identical_plans() -> None:
    candidates = (
        TeacherCandidate("a", np.array([0.1, 0.5]), 1.0, 0.04, 0.5, 0.3, "a"),
        TeacherCandidate("b", np.array([0.8, 0.5]), 0.5, 0.01, 0.4, 0.1, "b"),
        TeacherCandidate("c", np.array([0.9, 0.5]), 0.5, 0.02, 0.8, 0.2, "c"),
    )

    labels = select_teacher_labels(candidates)

    assert {label.plan_hash for label in labels} == {"b", "c"}
    assert next(label for label in labels if label.plan_hash == "b").teacher_types == (
        "COST", "BALANCE"
    )
    assert next(label for label in labels if label.plan_hash == "c").teacher_types == (
        "RELIABILITY",
    )
    assert all(np.array_equal(label.unbounded_action, np.log(label.scores / (1-label.scores))) for label in labels)
