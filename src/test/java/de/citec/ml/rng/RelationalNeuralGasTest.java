/* 
 * Relational Neural Gas
 * 
 * Copyright (C) 2015-2017
 * Benjamin Paaßen
 * AG Machine Learning
 * Centre of Excellence Cognitive Interaction Technology (CITEC)
 * University of Bielefeld
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package de.citec.ml.rng;

import java.util.Random;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public class RelationalNeuralGasTest {

	private static final double[] X = {1, 1.1, 3, 3.1, 3.2};
	private static RNGModelImpl mockModel;

	public RelationalNeuralGasTest() {
	}

	@BeforeClass
	public static void setUpClass() {
		// We create a mock dataset with two clusters, one close to 1 
		// (two datapoints) and one close to 3 (three datapoints).
		final int m = X.length;
		final double[][] D = new double[m][m];
		for (int i = 0; i < X.length; i++) {
			for (int j = 0; j < X.length; j++) {
				D[i][j] = Math.abs(X[i] - X[j]);
			}
		}
		final double[][] Alpha = {{0.6, 0.4, 0, 0, 0}, {0, 0, 0.5, 0.3, 0.2}};
		final int K = Alpha.length;
		final double[] Z = RelationalDistances.getNormalizationTerms(D, Alpha);
		final double[][] Dp = RelationalDistances.getDistancesToPrototypes(D, Alpha, Z);

		mockModel = new RNGModelImpl(Alpha, Dp, Z);
	}

	@AfterClass
	public static void tearDownClass() {
	}

	@Before
	public void setUp() {
	}

	@After
	public void tearDown() {
	}

	/**
	 * Test of train method, of class RelationalNeuralGas.
	 */
	@Test
	public void testTrain() {
		// We create a dataset of twodimensional points with three clusters.
		final int m = 90;
		final int K = 3;
		final int dims = 2;
		final double sigma = 0.1;
		final double[][] data = new double[m][2];
		final Random rand = new Random();
		for (int i = 0; i < m; i++) {
			int trueClusterIdx = i * K / m;
			// we generate the data as gaussian clusters, distributed along the
			// x axis
			data[i][0] = trueClusterIdx + sigma * rand.nextGaussian();
			for (int d = 1; d < dims; d++) {
				data[i][d] = sigma * rand.nextGaussian();
			}
		}
		// compute the dissimilarity matrix as squared Euclidean distances.
		final double[][] D = new double[m][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				for (int d = 0; d < dims; d++) {
					final double diff = data[i][d] - data[j][d];
					D[i][j] += diff * diff;
				}
			}
		}
		// then compute the clusters.
		final RNGErrorModel model = RelationalNeuralGas.train(D, K, 10);
		// first check the assignments.
		for (int k = 0; k < K; k++) {
			final int[] clusterMembers = RelationalNeuralGas.getClusterMembers(model, k);
			// we expect an even distribution of datapoints to clusters.
			assertEquals(m / 3, clusterMembers.length);
			// we expect that the first data point index is exactly divisible by
			// the cluster size.
			assertEquals(0, clusterMembers[0] * 3 % m);
			// and the other data points in the cluster should be direct successors.
			for (int j = 1; j < clusterMembers.length; j++) {
				assertEquals(clusterMembers[0] + j, clusterMembers[j]);
			}
		}
		// then check the position of the prototypes. We expect that the
		// prototypes should be fairly close to the center of the gaussian
		// clusters.
		for (int k = 0; k < K; k++) {
			final double[] alpha = model.getConvexCoefficients()[k];
			final double[] w = new double[dims];
			for (int i = 0; i < m; i++) {
				for (int d = 0; d < dims; d++) {
					w[d] += alpha[i] * data[i][d];
				}
			}
			assertEquals(0, w[0] - Math.round(w[0]), 2 * sigma);
			for (int d = 1; d < dims; d++) {
				assertEquals(0, w[d], 2 * sigma);
			}
		}
	}

	/**
	 * Test of getLambda method, of class RelationalNeuralGas.
	 */
	@Test
	public void testGetLambda() {
		// check that the initial lambda is K / 2, independent of the total number of epochs
		assertEquals(1, RelationalNeuralGas.getLambda(0, 2, 2), 1E-3);
		assertEquals(2, RelationalNeuralGas.getLambda(0, 4, 7), 1E-3);
		assertEquals(3, RelationalNeuralGas.getLambda(0, 6, 3), 1E-3);
		// check that the final lambda is always 0.01, independent of the number of prototypes
		assertEquals(0.01, RelationalNeuralGas.getLambda(6, 1, 7), 1E-3);
		assertEquals(0.01, RelationalNeuralGas.getLambda(7, 2, 8), 1E-3);
		assertEquals(0.01, RelationalNeuralGas.getLambda(0, 3, 1), 1E-3);
		// and check that lambda is monotonously decreasing for higher t
		final int T = 10;
		final int K = 4;
		double last_lambda = RelationalNeuralGas.getLambda(0, K, T);
		assertEquals(K / 2, last_lambda, 1E-3);

		for (int t = 1; t < T; t++) {
			final double current_lambda = RelationalNeuralGas.getLambda(t, K, T);
			assertTrue(current_lambda < last_lambda);
			last_lambda = current_lambda;
		}
		assertEquals(0.01, last_lambda, 1E-3);
	}

	/**
	 * Test of getAssignments method, of class RelationalNeuralGas.
	 */
	@Test
	public void testGetAssignments() {
		final int[] expected = {0, 0, 1, 1, 1};
		assertArrayEquals(expected, RelationalNeuralGas.getAssignments(mockModel));
	}

	/**
	 * Test of getClusterMembers method, of class RelationalNeuralGas.
	 */
	@Test
	public void testGetClusterMembers() {
		final int[][] expected = {
			{0, 1},
			{2, 3, 4}
		};
		final int[][] actual = RelationalNeuralGas.getClusterMembers(mockModel);
		assertEquals(expected.length, actual.length);
		for (int k = 0; k < expected.length; k++) {
			assertArrayEquals(expected[k], actual[k]);
		}
	}

	/**
	 * Test of getExamplars method, of class RelationalNeuralGas.
	 */
	@Test
	public void testGetExamplars() {
		final int[] expected = {0, 3};
		assertArrayEquals(expected, RelationalNeuralGas.getExamplars(mockModel));
	}

	/**
	 * Test of classify method, of class RelationalNeuralGas.
	 */
	@Test
	public void testClassify() {
		// check a few example data points in relation to the mock model above
		final double[] X_test = {1, -5, 1.5, 2.5, 3, 1000};
		final double[][] D = new double[X_test.length][X.length];
		for (int j = 0; j < X_test.length; j++) {
			for (int i = 0; i < X.length; i++) {
				D[j][i] = Math.abs(X_test[j] - X[i]);
			}
		}
		// we expect that all points smaller than 2 are put in the first and all points larger than
		// 2 are put in the second cluster
		final int[] expected = new int[X_test.length];
		for (int j = 0; j < X_test.length; j++) {
			expected[j] = X_test[j] < 2 ? 0 : 1;
		}

		assertArrayEquals(expected, RelationalNeuralGas.classify(D, mockModel));
	}

}
