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
 * This is an interface for the result of a Relational Neural Gas algorithm.
 * It encapsulates two fundamental properties of a RNG algorithm:
 * 1.) It assigns data points to clusters.
 * 2.) Each cluster has a prototype, which is represented as a convex combination.
 * We represent data points here only in terms of indices, numbered from 0 to
 * m-1, where m is the number of data points. Similarly, we represent clusters
 * as indices, from 0 to K-1, where K is the number of clusters/prototypes.
 * </p>
 *
 * <p>
 * As representation of prototypes, we choose the alpha-vector of convex
 * coefficients. Each prototype in the RNG framework is given as a convex
 * combination of the data points, where the coefficients are positive real
 * numbers that sum up to 1. Note, that there are techniques which make the
 * coefficient vectors sparse (e.g. median approaches, which select only one
 * alpha<sub>i</sub> = 1).
 * </p>
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public interface RNGModel {

	/**
	 * Returns the number of datapoints N.
	 *
	 * @return the number of datapoints N.
	 */
	public int getNumberOfDatapoints();

	/**
	 * Returns the number of clusters/prototypes K.
	 *
	 * @return the number of clusters/prototypes K.
	 */
	public int getNumberOfPrototypes();

	/**
	 * Returns the convex coefficients representing all prototypes. If m is the
	 * number of data points and K is the number of clusters/prototypes, this is
	 * a K x m matrix, where Alpha[k][i] represents the contribution of data point i to prototype
	 * k with Alpha[k][i] being always non-negative and the entries in Alpha[k] adding up to 1 for
	 * all k.
	 *
	 * @return the convex coefficients representing all prototypes.
	 */
	public double[][] getConvexCoefficients();

	/**
	 * Returns the matrix of distances from all training data points to all prototypes. For K
	 * prototypes and m datapoints, this is a K x m matrix Dp where entry Dp[i][k] contains the
	 * distance of data point i to prototype K.
	 *
	 * @return the matrix of distances from all prototypes to the data.
	 */
	public double[][] getDistancesToPrototypes();

	/**
	 * <p>
	 * Returns a K x 1 vector containing the normalization terms for for relational distances to
	 * each prototype. In particular, the distance of a data point x to prototype w<sub>k</sub>
	 * is given as
	 * </p>
	 *
	 * <p>
	 * d(w<sub>k</sub>, x)² = &Sigma;<sub>i=1,...,m</sub> &alpha;<sub>k, i</sub> &middot; d(x,
	 * x<sub>i</sub>)²
	 * - 0.5 &middot; &alpha;<sub>k</sub> &middot; D² &middot; &alpha;<sub>k</sub><sup>T</sup>
	 * </p>
	 *
	 * <p>
	 * where x<sub>1</sub>, ..., x<sub>m</sub> are the training data points and D is the matrix
	 * of pairwise distances between all training data points. The k-th entry of the returned vector
	 * contains exactly the term
	 * </p>
	 *
	 * <p>
	 * - 0.5 &middot; &alpha;<sub>k</sub> &middot; D² &middot; &alpha;<sub>k</sub><sup>T</sup>
	 * </p>
	 *
	 * @return a K x 1 vector containing the normalization terms for for relational distances to
	 * each prototype.
	 */
	public double[] getNormalizationTerms();

}
