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

/**
 * This is a class for consistency checks on input parameters.
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public final class CheckFunctions {

	private CheckFunctions() {

	}

	/**
	 * This is the tolerance for
	 */
	private static final double DOUBLE_TOLERANCE = 1E-8;

	/**
	 * Checks if the given matrix is quadratic, symmetric and reflexive. If it does not meet one of
	 * the criteria, this methods returns an exception accordingly. If all criteria are met, this
	 * method returns null.
	 *
	 * @param D supposedly a distance matrix.
	 *
	 * @return null if the matrix is a proper distance matrix, and an
	 * exception if it is not.
	 */
	public static IllegalArgumentException checkDissimilarityMatrix(final double[][] D) {
		if (D == null) {
			return new IllegalArgumentException(new NullPointerException("Input matrix is null."));
		}
		// first check if it is a square matrix.
		final int m = D.length;
		if (m == 0) {
			return new IllegalArgumentException("The input matrix is empty.");
		}
		for (int i = 0; i < m; i++) {
			if (D[i].length != m) {
				return new IllegalArgumentException("In row " + i
						+ " the input matrix has " + D[i].length
						+ " columns, but we expected " + m + " columns.");
			}
		}
		// compute te mean of the matrix elements to have an adjusted tolerance level for the
		// reflexivity and symmetry check
		double sum = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				sum += D[i][j];
			}
		}
		final double mean = sum / (double) (m * m);
		final double tolerance = mean * DOUBLE_TOLERANCE;
		// check reflexivity.
		for (int i = 0; i < m; i++) {
			if (D[i][i] > tolerance) {
				return new IllegalArgumentException("The given matrix is not "
						+ "reflective: Entry (" + i + ", " + i
						+ ") is larger than zero."
				);
			}
		}
		// check symmetry.
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < m; j++) {
				if (Math.abs(D[i][j] - D[j][i]) > tolerance) {
					return new IllegalArgumentException("The given matrix is "
							+ "not symmetric: Entry (" + i + ", " + j
							+ ") and entry (" + j + ", " + i + ") do not equal.");
				}
			}
		}
		return null;
	}

	/**
	 * Checks if the given matrix contains proper convex coefficient vectors.
	 *
	 * @param m the expected length of the convex combination.
	 * @param K the expected number of clusters/prototypes.
	 * @param Alpha supposedly convex combinations.
	 *
	 * @return null if Alpha contains proper convex combinations, and an
	 * exception if it does not.
	 */
	public static IllegalArgumentException checkConvexCoefficients(final int m, final int K, final double[][] Alpha) {
		if (Alpha == null) {
			return new IllegalArgumentException(new NullPointerException("Input matrix is null."));
		}
		if (Alpha.length != K) {
			return new IllegalArgumentException("Expected a convex combination "
					+ "for each of the " + K + " prototypes, but the given "
					+ "matrix has " + Alpha.length + " rows.");
		}
		for (int k = 0; k < K; k++) {
			final IllegalArgumentException ex = checkConvexCoefficients(m, Alpha[k]);
			if (ex != null) {
				return new IllegalArgumentException("The given vector for "
						+ "prototype " + k + " is not a proper convex "
						+ "combination.", ex);
			}
		}
		return null;
	}

	/**
	 * Checks if the given vector has the correct size (N), is non-negative and sums up to 1.
	 *
	 * @param m the expected length of the convex combination.
	 * @param alpha supposedly a convex combination.
	 *
	 * @return null if alpha is a proper convex combination, and an
	 * exception if it is not.
	 */
	public static IllegalArgumentException checkConvexCoefficients(final int m, final double[] alpha) {
		if (alpha == null) {
			return new IllegalArgumentException(new NullPointerException("Input vector is null."));
		}
		if (alpha.length != m) {
			return new IllegalArgumentException("Expected a convex combination "
					+ "for " + m + " data points, but the given vector had "
					+ alpha.length + " entries.");
		}
		// check non-negativity.
		double sum = 0;
		for (int i = 0; i < m; i++) {
			if (alpha[i] < 0) {
				return new IllegalArgumentException("Entry " + i + " of the "
						+ "given vector is negative.");
			}
			sum += alpha[i];
		}
		// check sum
		if (Math.abs(sum - 1.) > DOUBLE_TOLERANCE) {
			return new IllegalArgumentException("The given vector does not sum "
					+ "up to 1 but to " + sum);
		}
		return null;
	}

	/**
	 * Checks if the given assignment vector does indeed only map to entries
	 * between 0 and C, where C is the number of clusters/prototypes.
	 *
	 * @param N the expected length of the assignments vector.
	 * @param K the expected number of clusters/prototypes.
	 * @param assignments supposedly an assignments vector.
	 *
	 * @return null if the input is a proper assignments vector, and an
	 * exception if it is not.
	 */
	public static IllegalArgumentException checkAssignmentsVector(int N, int K, int[] assignments) {
		if (assignments == null) {
			return new IllegalArgumentException(new NullPointerException("Input vector is null."));
		}
		if (assignments.length != N) {
			return new IllegalArgumentException("Expected an assignments vector "
					+ "for " + N + " data points, but the given vector had "
					+ assignments.length + " entries.");
		}
		// gather all entries.
		for (int i = 0; i < N; i++) {
			if (assignments[i] < 0 || assignments[i] >= K) {
				return new IllegalArgumentException("Expected " + K
						+ " clusters, but data point " + i + " is assigned to"
						+ " cluster" + assignments[i]);
			}
		}
		return null;
	}

	/**
	 * Checks if the given matrix is not empty and consistent.
	 *
	 * @param Dp a matrix of distances from data points to prototypes.
	 *
	 * @return null if the input is a proper  matrix and an
	 * exception if it is not.
	 */
	public static IllegalArgumentException checkDissimilaritiesOfDatapointsToPrototypes(double[][] Dp) {
		if (Dp == null) {
			return new IllegalArgumentException(new NullPointerException("Input matrix is null."));
		}
		// first check if it is a consistent matrix.
		final int K = Dp.length;
		if (K == 0) {
			return new IllegalArgumentException("The input matrix is empty.");
		}
		final int m = Dp[0].length;
		for (int i = 1; i < K; i++) {
			if (Dp[i].length != m) {
				return new IllegalArgumentException("In row " + i
						+ " the input matrix has " + Dp[i].length
						+ " columns, but we expected " + m + " columns.");
			}
		}
		return null;
	}

}
