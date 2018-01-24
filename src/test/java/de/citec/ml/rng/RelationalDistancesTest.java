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
public class RelationalDistancesTest {

	public RelationalDistancesTest() {
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
	 * Test of getDistancesToPrototypes method, of class RelationalDistances.
	 */
	@Test
	public void testGetDistancesToPrototypes() {
		final double[] X = {-1, 0, 1};
		// create a simple 3 x 3 distance matrix
		final double[][] D = {
			{0, 1, 2},
			{1, 0, 1},
			{2, 1, 0}
		};
		// now, consider different prototypes; three sparse ones which select just one
		// of the data points, and two mixed ones
		final double[][] Alpha = {
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
			{0.5, 0.5, 0},
			{1. / 3, 1. / 3, 1. / 3}
		};
		// compute the expected distances from all data points to all prototypes. We expect that
		// these should be equal to the distances in the underlying Euclidean space
		final int m = X.length;
		final int K = Alpha.length;
		// compute the prototype locations first
		final double[] W = new double[K];
		for (int k = 0; k < K; k++) {
			for (int i = 0; i < m; i++) {
				W[k] += Alpha[k][i] * X[i];
			}
		}
		// then compute the expected squared distances
		final double[][] expected = new double[m][K];
		for (int i = 0; i < m; i++) {
			for (int k = 0; k < K; k++) {
				expected[i][k] = (X[i] - W[k]) * (X[i] - W[k]);
			}
		}
		// compute with the actual result
		final double[] Z = RelationalDistances.getNormalizationTerms(D, Alpha);
		final double[][] actual = RelationalDistances.getDistancesToPrototypes(D, Alpha, Z);

		for (int i = 0; i < m; i++) {
			assertArrayEquals(expected[i], actual[i], 1E-3);
		}
	}

	/**
	 * Test of getNormalizationTerms method, of class RelationalDistances.
	 */
	@Test
	public void testGetNormalizationTerms() {
		// create a simple 3 x 3 distance matrix
		final double[][] D = {
			{0, 1, 2},
			{1, 0, 1},
			{2, 1, 0}
		};
		// now, consider different prototypes; three sparse ones which select just one
		// of the data points, and two mixed ones
		final double[][] Alpha = {
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
			{0.5, 0.5, 0},
			{1. / 3, 1. / 3, 1. / 3}
		};
		// for the sparse vectors, the normalization terms should be zero; for all others,
		// we expect nonzero values as follows
		final double[] expected = new double[Alpha.length];
		for (int k = 3; k < Alpha.length; k++) {
			for (int i = 0; i < D.length; i++) {
				for (int j = 0; j < D.length; j++) {
					expected[k] += -0.5 * Alpha[k][i] * Alpha[k][j] * D[i][j] * D[i][j];
				}
			}
		}

		final double[] actual = RelationalDistances.getNormalizationTerms(D, Alpha);
		assertArrayEquals(expected, actual, 1E-3);
	}

}
