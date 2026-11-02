"""Run the study guide's small numerical examples. No model or data downloads."""
import numpy as np


def main():
    a = np.outer([1, 2, 3], [1, 2, 3])
    b = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 2]])
    assert np.linalg.matrix_rank(a) == 1
    assert np.linalg.matrix_rank(b) == 2
    np.testing.assert_array_equal(b @ [-1, -1, 1], [0, 0, 0])
    m = np.array([[3., 1.], [1., 3.]])
    u, s, vt = np.linalg.svd(m)
    np.testing.assert_allclose(u @ np.diag(s) @ vt, m)
    m1 = s[0] * np.outer(u[:, 0], vt[0])
    np.testing.assert_allclose(m1, np.full((2, 2), 2.))
    np.testing.assert_allclose(np.linalg.norm(m - m1), 2.)
    # Rank-one correction to an identity matrix does not make the whole matrix rank one.
    correction = np.outer([1., 0., 0.], [0., 1., 0.])
    assert np.linalg.matrix_rank(correction) == 1
    assert np.linalg.matrix_rank(np.eye(3) + correction) == 3
    full, low_rank = 384 * 384, 8 * (384 + 384)
    print("Ranks:", np.linalg.matrix_rank(a), np.linalg.matrix_rank(b))
    print("Singular values:", s)
    print("Rank-one approximation:\n", m1)
    print("Frobenius error:", np.linalg.norm(m - m1))
    print("Retained squared singular-value fraction:", s[0] ** 2 / np.sum(s ** 2))
    print(f"One projection: full={full}, rank-8 LoRA={low_rank} ({100*low_rank/full:.2f}%)")
    print("Q/V in 12 blocks:", 12 * 2 * low_rank, "adapter parameters")
    print("Rank-one update + rank-three base can remain rank three. All checks passed.")


if __name__ == "__main__":
    main()
