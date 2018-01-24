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
 * <p>
 * This is a helper class for the computation of distances between data points and convex
 * combinations. As demonstrated in Hasenfuss and Hammer (2009), if a prototype w<sub>k</sub> is
 * given in terms of a convex combination
 * </p>
 *
 * <p>
 * w<sub>k</sub> = &Sigma;<sub>i=1,...,m</sub> &alpha;<sub>k, i</sub> &middot; x<sub>i</sub>
 * </p>
 *
 * <p>
 * were all convex coefficients &alpha;<sub>k, i</sub> are non-negative and
 * </p>
 *
 * <p>
 * &Sigma;<sub>i=1,...,m</sub> &alpha;<sub>k, i</sub> = 1
 * </p>
 *
 * <p>
 * we obtain for all i:
 * </p>
 *
 * <p>
 * d(w<sub>k</sub>, x<sub>i</sub>)² = &alpha;<sub>k</sub> &middot; D(:, i)² - 0.5 &middot;
 * &alpha;<sub>k</sub> &middot; D² &middot; &alpha;<sub>k</sub><sup>T</sup>
 * </p>
 *
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public final class RelationalDistances {

	private RelationalDistances() {

	}

	/**
	 * <p>
	 * Calculates the squared distances of n data points to all prototypes, based on the
	 * distances from the test to the training data D and a relational neural gas model.
	 * </p>
	 *
	 * @param D The n x m distance matrix from test data points to training data points.
	 * @param model a RNGModel.
	 *
	 * @return a n x K matrix Dp where Dp[j][k] contains the squared distance between data point
	 * j and prototype k.
	 */
	public static double[][] getDistancesToPrototypes(double[][] D, RNGModel model) {
		return getDistancesToPrototypes(D, model.getConvexCoefficients(), model.getNormalizationTerms());
	}

	/**
	 * <p>
	 * Calculates the squared distances of a data point to all prototypes, based on the
	 * distances to the training data d and a relational neural gas model.
	 * </p>
	 *
	 * @param d The 1 x m vector of distances from the data point to all training data points.
	 * @param model a RNGModel.
	 *
	 * @return a 1 x K vector dp where dp[k] contains the squared distance between the data point
	 * and prototype k.
	 */
	public static double[] getDistancesToPrototypes(double[] d, RNGModel model) {
		return getDistancesToPrototypes(d, model.getConvexCoefficients(), model.getNormalizationTerms());
	}

	/**
	 * <p>
	 * Calculates the squared distances of n data points to all prototypes, based on the
	 * distances from the test to the training data D, the normalization terms for each prototype
	 * Z, and the convex coefficients representing the prototypes Alpha. In particular, let m
	 * be the number of training data points and K be the number of prototypes. Then, the output
	 * is a n x K matrix Dp, where
	 * </p>
	 *
	 * <p>
	 * Dp[j][k] = &Sigma;<sub>i=1,...,m</sub> Alpha[k][i] &middot; D[j][i]² + Z[k]
	 * - 0.5 * Alpha[k] &middot; D_train² &middot; Alpha[k]<sup>T</sup>
	 * </p>
	 *
	 * @param D The n x m distance matrix from test data points to training data points.
	 * @param Alpha The prototypes given as a K x m matrix of convex coefficients.
	 * @param Z The K x 1 vector of normalization terms for each prototype, meaning prototype k
	 * contains
	 * - 0.5 * Alpha[k] &middot; D_train² &middot; Alpha[k]<sup>T</sup>
	 * where D_train is the matrix of pairwise distances for the training data points.
	 *
	 * @return a n x K matrix Dp where Dp[j][k] contains the squared distance between data point
	 * j and prototype k.
	 */
	protected static double[][] getDistancesToPrototypes(double[][] D, double[][] Alpha, double[] Z) {

		final IllegalArgumentException ex = CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(D);
		if (ex != null) {
			throw ex;
		}
		final int n = D.length;
		final double[][] Dp = new double[n][];
		for (int j = 0; j < n; j++) {
			Dp[j] = RelationalDistances.getDistancesToPrototypes(D[j], Alpha, Z);
		}
		return Dp;
	}

	/**
	 * <p>
	 * Calculates the squared distances of data point to all prototypes, based on the
	 * distances to the training data d, the normalization terms for each prototype
	 * Z, and the convex coefficients representing the prototypes Alpha. In particular, let m
	 * be the number of training data points and K be the number of prototypes. Then, the output
	 * is a 1 x K vector dp, where</p>
	 *
	 * <p>
	 * dp[k] = &Sigma;<sub>i=1,...,m</sub> Alpha[k][i] &middot; d[i]² + Z[k]
	 * - 0.5 * Alpha[k] &middot; D_train² &middot; Alpha[k]<sup>T</sup>
	 * </p>
	 *
	 * @param d The 1 x m vector of distances from the data point to all training data points.
	 * @param Alpha The prototypes given as a K x m matrix of convex coefficients.
	 * @param Z The K x 1 vector of normalization terms for each prototype, meaning prototype k
	 * contains
	 * - 0.5 * Alpha[k] &middot; D_train² &middot; Alpha[k]<sup>T</sup>
	 * where D_train is the matrix of pairwise distances for the training data points.
	 *
	 * @return a 1 x K vector dp where dp[k] contains the squared distance between the data point
	 * and prototype k.
	 */
	protected static double[] getDistancesToPrototypes(double[] d, double[][] Alpha, double[] Z) {
		final int m = d.length;
		final int K = Alpha.length;
		final double[] dp = new double[K];
		for (int k = 0; k < K; k++) {
			for (int i = 0; i < m; i++) {
				dp[k] += Alpha[k][i] * d[i] * d[i];
			}
			dp[k] += Z[k];
		}
		return dp;
	}

	/**
	 * <p>
	 * Computes the normalization terms
	 * </p>
	 *
	 * <p>
	 * -0.5 * Alpha[k] &middot; D² &middot; Alpha[k]<sup>T</sup>
	 * </p>
	 *
	 * <p>
	 * for all prototypes k, where Alpha[k] are the convex coefficients representing prototype k
	 * and D² is the matrix of squared distances for all training data points.
	 * </p>
	 *
	 * @param D a m x m matrix of distances for all training data points.
	 * @param Alpha The prototypes given as a K x m matrix of convex coefficients.
	 *
	 * @return a K x 1 vector containing the entries
	 * -0.5 * Alpha[k] &middot; D² &middot; Alpha[k]<sup>T</sup>
	 */
	protected static double[] getNormalizationTerms(double[][] D, double[][] Alpha) {
		final int K = Alpha.length;
		final int m = D.length;
		final double[] Z = new double[K];
		for (int k = 0; k < K; k++) {
			for (int i = 0; i < m; i++) {
				if (Alpha[k][i] == 0) {
					continue;
				}
				for (int j = 0; j < m; j++) {
					Z[k] += Alpha[k][i] * D[i][j] * D[i][j] * Alpha[k][j];
				}
			}
			Z[k] *= -0.5;
		}
		return Z;
	}

}
