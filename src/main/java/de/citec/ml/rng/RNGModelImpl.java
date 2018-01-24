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
 * This provides a default implementation of the RNGModel interface.
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public class RNGModelImpl implements RNGErrorModel {

	private final double[][] Alpha;
	private final double[][] Dp;
	private final double[] Z;
	private final double[] errors;

	/**
	 * Initializes an Relational Neural Gas model. Let m be the number of data
	 * points and K be the number of clusters, then:
	 *
	 * @param Alpha is a K x m matrix, where Alpha[k][i] represents the contribution of data point i
	 * to prototype k with Alpha[k][i] being always non-negative and the entries in Alpha[k] adding
	 * up to 1 for all k.
	 * @param Dp is a m x K matrix, where each entry Dp[i][k] contains the
	 * distance of data point i to prototype K.
	 * @param Z is a K x 1 vector where the k-th entry contains the term
	 * - 0.5 &middot; Alpha[k] &middot; D² &middot; Alpha[k]<sup>T</sup>
	 * where D² is the matrix of squared pairwise distances in the training data.
	 */
	public RNGModelImpl(double[][] Alpha, double[][] Dp, double[] Z) {
		this(Alpha, Dp, Z, null);
	}

	/**
	 * Initializes an Relational Neural Gas model. Let m be the number of data
	 * points, K be the number of clusters and E be the number of training
	 * epochs, then:
	 *
	 * @param Alpha is a K x m matrix, where Alpha[k][i] represents the contribution of data point i
	 * to prototype k with Alpha[k][i] being always non-negative and the entries in Alpha[k] adding
	 * up to 1 for all k.
	 * @param Dp is a m x K matrix, where each entry Dp[i][k] contains the
	 * distance of data point i to prototype K.
	 * @param Z is a K x 1 vector where the k-th entry contains the term
	 * - 0.5 &middot; Alpha[k] &middot; D² &middot; Alpha[k]<sup>T</sup>
	 * where D² is the matrix of squared pairwise distances in the training data.
	 * @param errors is an E x 1 vector, where each entry errors[e] contains the
	 * quantization error before training epoch e.
	 */
	public RNGModelImpl(double[][] Alpha, double[][] Dp, double[] Z, double[] errors) {

		// check the distance matrix.
		IllegalArgumentException ex = CheckFunctions.checkDissimilaritiesOfDatapointsToPrototypes(Dp);
		if (ex != null) {
			throw ex;
		}
		this.Dp = Dp;
		final int m = Dp.length;
		final int K = Dp[0].length;
		// check convex combinations.
		ex = CheckFunctions.checkConvexCoefficients(m, K, Alpha);
		if (ex != null) {
			throw ex;
		}
		this.Alpha = Alpha;
		if (Z == null || Z.length != K) {
			throw new IllegalArgumentException("Expected a " + K + " x 1 vector of normalization terms as third constructor argument!");
		}
		this.Z = Z;
		this.errors = errors;
	}

	@Override
	public int getNumberOfDatapoints() {
		return Dp.length;
	}

	@Override
	public int getNumberOfPrototypes() {
		return Alpha.length;
	}

	@Override
	public double[][] getDistancesToPrototypes() {
		return Dp;
	}

	@Override
	public double[] getNormalizationTerms() {
		return Z;
	}

	@Override
	public double[][] getConvexCoefficients() {
		return Alpha;
	}

	@Override
	public int getNumberOfEpochs() {
		return errors.length;
	}

	@Override
	public double[] getQuantizationErrors() {
		return errors;
	}

}
