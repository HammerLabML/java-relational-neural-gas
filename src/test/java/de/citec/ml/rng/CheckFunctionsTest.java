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

import de.citec.ml.rng.CheckFunctions;
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
public class CheckFunctionsTest {

	public CheckFunctionsTest() {
	}

	@BeforeClass
	public static void setUpClass() {
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
	 * Test of checkDissimilarityMatrix method, of class CheckFunctions.
	 */
	@Test
	public void testCheckDissimilarityMatrix() {
		// check a null matrix.
		assertNotNull(CheckFunctions.checkDissimilarityMatrix(null));
		// check a non-square matrix.
		assertNotNull(CheckFunctions.checkDissimilarityMatrix(new double[4][2]));
		// check an empty matrix.
		assertNotNull(CheckFunctions.checkDissimilarityMatrix(new double[0][0]));

		// check a non-reflexive matrix
		{
			final double[][] D = new double[4][4];
			D[2][2] = 0.01;
			assertNotNull(CheckFunctions.checkDissimilarityMatrix(D));
		}
		// check a non-symmetric matrix.
		{
			final double[][] D = new double[4][4];
			D[2][1] = 0.01;
			assertNotNull(CheckFunctions.checkDissimilarityMatrix(D));
		}
		// check a valid matrix (albeit a strange one, only containing zeros)
		assertNull(CheckFunctions.checkDissimilarityMatrix(new double[4][4]));
		// check a non-trivial valid matrix.
		{
			final double[] X = {1, 2, 3, 5};
			final double[][] D = new double[X.length][X.length];
			for (int i = 0; i < X.length; i++) {
				for (int j = 0; j < X.length; j++) {
					D[i][j] = Math.abs(X[i] - X[j]);
				}
			}
			assertNull(CheckFunctions.checkDissimilarityMatrix(D));
		}
	}

	/**
	 * Test of checkConvexCoefficients method, of class CheckFunctions.
	 */
	@Test
	public void testCheckConvexCoefficients_3args() {
		// check null input
		assertNotNull(CheckFunctions.checkConvexCoefficients(0, 0, null));
		// check wrong length
		{
			final int K = 2;
			final int N = 3;
			double[][] Alpha = {{1, 0, 0}};
			assertNotNull(CheckFunctions.checkConvexCoefficients(N, K, Alpha));
			Alpha = new double[][]{{1, 0, 0}, {1}};
			assertNotNull(CheckFunctions.checkConvexCoefficients(N, K, Alpha));
		}
		// check negativity
		{
			final int K = 1;
			final int N = 3;
			double[][] Alpha = {{2, 0, -1}};
			assertNotNull(CheckFunctions.checkConvexCoefficients(N, K, Alpha));
		}
		// check sum
		{
			final int K = 1;
			final int N = 3;
			double[][] Alpha = {{0.1, 0.2, 0.3}};
			assertNotNull(CheckFunctions.checkConvexCoefficients(N, K, Alpha));
		}
		// check valid one
		{
			final int K = 2;
			final int N = 3;
			double[][] Alpha = {{0.8, 0.2, 0}, {0.1, 0.2, 0.7}};
			assertNull(CheckFunctions.checkConvexCoefficients(N, K, Alpha));
		}
	}

	/**
	 * Test of checkConvexCoefficients method, of class CheckFunctions.
	 */
	@Test
	public void testCheckConvexCoefficients_int_doubleArr() {
		// check null input
		assertNotNull(CheckFunctions.checkConvexCoefficients(0, null));
		// check wrong length
		assertNotNull(CheckFunctions.checkConvexCoefficients(2, new double[]{1}));
		// check negativity
		assertNotNull(CheckFunctions.checkConvexCoefficients(2, new double[]{2, -1}));
		// check sum
		assertNotNull(CheckFunctions.checkConvexCoefficients(2, new double[]{0.5, 0.3}));
		// check valid one
		assertNull(CheckFunctions.checkConvexCoefficients(2, new double[]{0.5, 0.5}));
	}

	/**
	 * Test of checkAssignmentsVector method, of class CheckFunctions.
	 */
	@Test
	public void testCheckAssignmentsVector() {
		// check null input
		assertNotNull(CheckFunctions.checkAssignmentsVector(0, 0, null));
		// check wrong length
		assertNotNull(CheckFunctions.checkAssignmentsVector(2, 1, new int[]{1}));
		// check wrong cluster numbering
		assertNotNull(CheckFunctions.checkAssignmentsVector(2, 2, new int[]{1, 2}));
		// check valid one
		assertNull(CheckFunctions.checkAssignmentsVector(2, 2, new int[]{0, 1}));
		assertNull(CheckFunctions.checkAssignmentsVector(2, 2, new int[]{0, 0}));
	}

	/**
	 * Test of checkDissimilaritiesOfPrototypesToData method, of class
	 * CheckFunctions.
	 */
	@Test
	public void testCheckDissimilaritiesOfPrototypesToData() {
		// check a null matrix.
		assertNotNull(CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(null));
		// check an empty matrix.
		assertNotNull(CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(new double[0][0]));
		// check an inconsistent matrix.
		{
			final double[][] Dp = new double[2][];
			Dp[0] = new double[3];
			Dp[1] = new double[2];
			assertNotNull(CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(Dp));
		}

		// check a valid matrix
		assertNull(CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(new double[4][3]));
		// check a non-trivial valid matrix.
		{
			final double[] W = {1, 3};
			final double[] X = {0.8, 1.2, 2.7, 3.5};
			final double[][] Dp = new double[W.length][X.length];
			for (int k = 0; k < W.length; k++) {
				for (int j = 0; j < X.length; j++) {
					Dp[k][j] = Math.abs(W[k] - X[j]);
				}
			}
			assertNull(CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(Dp));
		}

	}

}
