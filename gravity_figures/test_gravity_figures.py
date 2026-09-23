"""Synthetic numerical/selection tests; run: python -m unittest -v."""
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gravity_common import (HARMONICS, SATELLITES, FULL_NAME, BRANCH, RL05,
                            load_solution, constellation_files)
from plot_field_rms import compute_rms


def write_solution(path, harmonics=HARMONICS, offset=0.0, covariance=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, h in enumerate(harmonics):
        family, nm = h.split('_')
        for epoch in ('20000101', '20000131', '20000301'):
            records.append((f'G{family}N {nm[0]} {nm[1]} {epoch}', (i+1)*1e-7 + offset, 1e-12))
    records += [('LAM', 1., 0.), ('LQM', .2, .02), ('LTM', .3, .03)]
    for h in harmonics:
        family, nm = h.split('_')
        family = 'P' if family == 'S' else 'C'
        for mode in ('B', 'A', 'AA', 'C', 'S'):
            records.append((family+mode+'_'+nm, 1e-11, 1e-13))
    n = len(records)-1
    if covariance is None:
        # Positive definite, nontrivial signed correlations; stable across runs.
        rng = np.random.default_rng(31)
        a = rng.normal(size=(n, n))
        covariance = a @ a.T + np.eye(n)
    with path.open('w') as f:
        f.write(f'HEADER\nSOLUTION\n{len(records):10d}{n:10d}{1:10d}\n')
        for name, value, sigma in records:
            f.write(f'{name:<24}{0.:24.14E}{value:24.14E}{value:24.14E}{sigma:24.14E}\n')
        f.write(f'INVERSE MATRIX\n{100:10d}{n:10d}{1.:20.14E}\n')
        # Width-20 Fortran-style fields, adjacent, with rows wrapped at six.
        for i in range(n):
            fields = []
            for value in covariance[i, :i+1]:
                # Formatter with a leading decimal mantissa fits 20 characters.
                mantissa, exponent = f'{abs(value):.13E}'.split('E')
                digits = mantissa.replace('.', '')
                token = ('-' if value < 0 else '0') + '.' + digits + 'E' + f'{int(exponent)+1:+03d}'
                assert len(token) == 20, token
                fields.append(token)
            for start in range(0, len(fields), 6):
                f.write(''.join(fields[start:start+6])+'\n')
    return covariance


def write_reference(path):
    with Path(path).open('w') as f:
        for n, m in ((2, 0), (2, 1), (4, 0), (4, 1), (6, 0)):
            c = (HARMONICS.index(f'C_{n}{m}')+1)*1e-7
            s = (HARMONICS.index(f'S_{n}{m}')+1)*1e-7 if m else 0.
            f.write(f'G_BIAS {n} {m} {c:.16e} {s:.16e} 2e-12 3e-12 20000101.0 20010101.0 0\n')


class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_packed_matrix_selection_and_fixed_exclusion(self):
        path = self.root/'result'
        matrix = write_solution(path)
        solution = load_solution(path, True)
        active = [p for p in solution.parameters if p.sigma]
        indices = [next(i for i, p in enumerate(active) if p.name == q.name)
                   for q in solution.correlation_parameters]
        d = np.sqrt(np.diag(matrix))
        expected = (matrix/d[:, None]/d[None, :])[np.ix_(indices, indices)]
        np.testing.assert_allclose(solution.correlation, expected, atol=1e-12)
        self.assertNotIn('LAM', [p.name for p in solution.correlation_parameters])
        self.assertTrue(all(p.date is None for p in solution.correlation_parameters))
        self.assertEqual(solution.series('C_20')[0].size, 3)
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1])+'\n')
        with self.assertRaisesRegex(ValueError, 'incomplete inverse matrix'):
            load_solution(path, True)

    def test_rl05_legacy_basis_uncertainty_and_interval_boundary(self):
        path = self.root/'RL05.shc'
        path.write_text('G_BIAS 2 0 10 0 2 0 20000101.0 20010101.0 0\n'
                        'G_DRIFT 2 0 3 0 4 0 20000101.0 20010101.0 0\n'
                        'G_COS 2 0 5 0 6 0 20000101.0 20010101.0 0\n'
                        'G_SIN 2 0 7 0 8 0 20000101.0 20010101.0 0\n')
        model = RL05(path)
        dates = np.array(['2000-01-01', '2000-07-01', '1999-01-01'], dtype='datetime64[D]')
        values, sigma = model.evaluate('C_20', dates)
        t = 182/365
        c, s = np.cos(2*np.pi*t), np.sin(2*np.pi*t)
        np.testing.assert_allclose(values[:2], [15, 10+3*t+5*c+7*s])
        np.testing.assert_allclose(sigma[:2], [8, 2+4*t+6*c+8*s])
        self.assertEqual(values[2], 0.)
        self.assertEqual(sigma[2], 0.)
        path.write_text('G_BIAS 2 0 1 0 2 0 20000101.0 20010101.0 0\n'
                        'G_BIAS 2 0 2 0 3 0 20010101.0 20020101.0 0\n')
        values, sigma = RL05(path).evaluate('C_20', ['2001-01-01'])
        np.testing.assert_equal(values, [3.])
        np.testing.assert_equal(sigma, [5.])

    def test_31_constellations_exact_tides_and_known_rms(self):
        reference = self.root/'RL05.shc'
        write_reference(reference)
        source = self.root/'solution_all_constellations'
        for size in range(1, 6):
            for constellation in combinations(SATELLITES, size):
                name = 'rheology_'+'_'.join(constellation)
                write_solution(source/(name+'_tides'), offset=size*1e-10)
                (source/(name+'_solid_tides')).write_text('must not be read')
        hidden = source/BRANCH/(FULL_NAME+'_tides')
        hidden.parent.mkdir(parents=True)
        hidden.write_text('must not be read')
        files = constellation_files(source)
        self.assertEqual(len(files), 31)
        self.assertEqual([len(c) for c, _ in files], sorted(len(c) for c, _ in files))
        constellations, rms, counts = compute_rms(source, reference)
        expected = np.array([len(c)*1e-10 for c in constellations])[:, None]*np.ones((1, 7))
        np.testing.assert_allclose(rms, expected, atol=1e-21)
        np.testing.assert_equal(counts, np.full((31, 7), 3))


if __name__ == '__main__':
    unittest.main()
