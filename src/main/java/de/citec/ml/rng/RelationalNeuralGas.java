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

import java.util.ArrayList;
import java.util.Random;

/**
 * <p>
 * This class implements the batch relational neural gas algorithm, which tries to minimize the
 * soft quantization error for prototypes to data points and anneals the softness over time until
 * the algorithm converges to K-means. As such, relational neural gas can be seen as a proper
 * and more robust generalization of (relational) K-means.
 * </p>
 *
 * <p>
 * To be more precise, let x<sub>1</sub>, ..., x<sub>m</sub> be data points and
 * w<sub>1</sub>, ..., w<sub>K</sub> be the prototypes of neural gas. Then, neural gas minimizes the
 * error
 * </p>
 *
 * <p>
 * E = &Sigma;<sub>i=1,...,m</sub> &Sigma;<sub>k=1,...,K</sub>
 * h(k|i) &middot; d(x<sub>i</sub>, w<sub>k</sub>)²
 * </p>
 *
 * <p>
 * where h(k|i) is the assignment strength of data point i to prototype k. Neural gas models this
 * assignment strength as
 * </p>
 *
 * <p>
 * h(k|i) = exp(-r(k|i) / &lambda;)
 * </p>
 *
 * <p>
 * where r(k|i) is the rank of prototype k with respect to data point i, that is, the number of
 * other prototypes which are closer to data point i than prototype w<sub>k</sub>, and &lambda; is a
 * hyper-parameter which regulates the 'crispness' of the assignment. For &lambda; = 0,
 * every data point is assigned strictly to the closest prototype and to no other. In neural gas,
 * &lambda; starts at K/2 and is then annealed via the falling exponential equation
 * </p>
 *
 * <p>
 * &lambda;<sub>t</sub> := &lambda;<sub>0</sub> &middot; (0.01 / &lambda;<sub>0</sub>)<sup>(t / T)</sup>
 * </p>
 *
 * <p>
 * where t is the current epoch and T is the overall number of epochs. So &lambda;<sub>T</sub> is
 * exactly 0.01.</p>
 *
 * <p>
 * The special property of relational neural gas is that d(x<sub>i</sub>, w<sub>k</sub>)
 * is not computed via the Euclidean distance but is computed indirectly. In particular, we
 * assume that prototype w<sub>k</sub> is given as a convex combination of the form</p>
 *
 * <p>
 * w<sub>k</sub> = &Sigma;<sub>i=1,...,m</sub> &alpha;<sub>k, i</sub> &middot; x<sub>i</sub>
 * </p>
 *
 * <p>
 * and that we have a m x m matrix of pairwise distances D between all data points available. In
 * This case, the distance between prototypes and data points is given as:</p>
 *
 * <p>
 * d(w<sub>k</sub>, x<sub>i</sub>)² = &alpha;<sub>k</sub> &middot; D(:, i)² - 0.5 &middot;
 * &alpha;<sub>k</sub> &middot; D² &middot; &alpha;<sub>k</sub><sup>T</sup>
 * </p>
 *
 * <p>
 * where &alpha;<sub>k</sub> is the vector of convex coefficients for prototype k and D(:, i)²
 * is the i-th column of the squared distance matrix D².
 * </p>
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public final class RelationalNeuralGas {

	private RelationalNeuralGas() {

	}
	/**
	 * If an exponent x is bigger than this threshold, we approximate exp(-x) =
	 * 0.
	 */
	private static final double APPROX_THRESHOLD = -Math.log(1E-3);

	/**
	 * Trains a relational neural gas model with K prototypes for the given data in terms of a m x m
	 * matrix of pairwise distances D, using 30 iterations of the RNG algorithm.
	 *
	 * The algorithm takes O(m &middot; m &middot; K) calculation steps.
	 *
	 * @param D a m x m distance matrix. Note that this matrix is required to be <em>symmetric</em>
	 * and have a zero diagonal.
	 * @param K the number of prototypes for the relational neural gas model.
	 *
	 * @return a relational neural gas model.
	 */
	public static RNGErrorModel train(double[][] D, int K) {
		return train(D, K, 30);
	}

	/**
	 * Trains a relational neural gas model with K prototypes for the given data in terms of a m x m
	 * matrix of pairwise distances D, using T iterations of the RNG algorithm.
	 *
	 * The algorithm takes O(m &middot; m &middot; K &middot; T) calculation steps.
	 *
	 * @param D a m x m distance matrix. Note that this matrix is required to be <em>symmetric</em>
	 * and have a zero diagonal.
	 * @param K the number of prototypes for the relational neural gas model.
	 * @param T the number of epochs used for training.
	 *
	 * @return a relational neural gas model.
	 */
	public static RNGErrorModel train(double[][] D, int K, int T) {
		// check input
		final IllegalArgumentException ex = CheckFunctions.checkDissimilarityMatrix(D);
		if (ex != null) {
			throw ex;
		}
		if (K < 1) {
			throw new IllegalArgumentException("The number of prototypes must be positive!");
		}
		if (T < 1) {
			throw new IllegalArgumentException("The number of epochs must be positive!");
		}

		final int m = D.length;
		// initialize the convex coefficients randomly at first.
		final double[][] Alpha = new double[K][m];
		{
			final Random rand = new Random();
			for (int k = 0; k < K; k++) {
				double nrml = 0;
				for (int i = 0; i < m; i++) {
					Alpha[k][i] = rand.nextDouble();
					nrml += Alpha[k][i];
				}
				for (int i = 0; i < m; i++) {
					Alpha[k][i] /= nrml;
				}
			}
		}
		final double[] errors = new double[T + 1];
		// start the learning algorithm
		for (int t = 0; t < T; t++) {
			// retrieve the lambda parameter for this epoch.
			final double invLambda = 1. / getLambda(t, K, T);
			// generate the possible exponential values.
			final double[] hs = new double[K];
			hs[0] = 1;
			int max_rank = 1;
			for (; max_rank < K; max_rank++) {
				final double exponent = invLambda * max_rank;
				// approximate the exponential decay for large ranks with 0.
				if (exponent > APPROX_THRESHOLD) {
					break;
				}
				hs[max_rank] = Math.exp(-exponent);
			}

			// calculate the current normalization terms for all prototypes
			final double[] Z = RelationalDistances.getNormalizationTerms(D, Alpha);

			// calculate the distances of all datapoints to all prototypes.
			final double[][] Dp = RelationalDistances.getDistancesToPrototypes(D, Alpha, Z);
			// calculate the ranking of data points to prototypes (from the
			// perspective of some data point: Which prototypes are the closest?)
			// Note that we only need to compute the ranking up to K_effective, because
			// all data prototypes beyond that contribute nothing.
			// We compute the ranking via insertion sort
			// From the ranking, we compute the data point to prototype assignment matrix H.
			final double[][] H = new double[K][m];
			for (int i = 0; i < m; i++) {
				final int[] ranking = new int[max_rank];
				for (int k = 1; k < K; k++) {
					// only insert this prototype into the existing ranking if it is better than
					// the least ranked prototype
					if (k < max_rank || Dp[i][k] < Dp[i][ranking[max_rank - 1]]) {
						// first consider the last rank as insertion point
						int rank = k < max_rank ? k : max_rank - 1;
						// then check whether the distance to prototype k is actually smaller than
						// prior ranks; if so, decrease the rank
						while (rank > 0 && Dp[i][k] < Dp[i][ranking[rank - 1]]) {
							if (rank < max_rank) {
								ranking[rank] = ranking[rank - 1];
							}
							rank--;
						}
						// if the rank is below K_effective, insert the prototype index into the rank
						// matrix.
						if (rank < max_rank) {
							ranking[rank] = k;
						}
					}
				}
				// once the ranking is computed, use it to compute the H matrix
				for (int r = 0; r < max_rank; r++) {
					H[ranking[r]][i] = hs[r];
				}
			}

			// update the prototype locations based on the data-point to prototype assignments.
			// in particular, our new prototype location is
			// W[k] = sum over i H[k][i] * x[i] / sum over i H[k][i]
			for (int k = 0; k < K; k++) {
				// calculate the sum over all H terms for this prototype
				double nrml = 0;
				for (int i = 0; i < m; i++) {
					if (H[k][i] == 0) {
						continue;
					}
					nrml += H[k][i];
					// track the quantization error
					errors[t] += H[k][i] * Dp[i][k];
				}
				// then calculate the alpha-updates
				for (int i = 0; i < m; i++) {
					Alpha[k][i] = H[k][i] / nrml;
				}
			}
		}
		// After the training, compute the normalization terms and the distances from data points
		// to prototypes one more time
		final double[] Z = RelationalDistances.getNormalizationTerms(D, Alpha);
		final double[][] Dp = RelationalDistances.getDistancesToPrototypes(D, Alpha, Z);
		// and calculate the crisp quantization error as final error
		for (int i = 0; i < m; i++) {
			final int k = ArrayFunctions.getMinIdx(Dp[i]);
			errors[T] += Dp[i][k];
		}

		// Then, return the final model.
		return new RNGModelImpl(Alpha, Dp, Z, errors);
	}

	/**
	 * <p>
	 * Returns the &lambda; value for the current training epoch. As recommended by Martinez et al.
	 * (1993; cited after Hasenfuss, 2009) we use an exponential decay (annealing) of &lambda; with
	 * </p>
	 *
	 * <p>
	 * &lambda;<sub>t</sub> := &lambda;<sub>0</sub> * (0.01 / &lambda;<sub>0</sub>)^(t / (T-1))
	 * </p>
	 *
	 * <p>
	 * where &lambda;<sub>0</sub> is the number of prototypes divided by 2, t is the current epoch and
	 * T is the total number of epochs.</p>
	 *
	 * @param t the current training epoch.
	 * @param K the number of prototypes.
	 * @param T the overall number of training epochs.
	 *
	 * @return the &lambda; value for the current training epoch.
	 */
	public static double getLambda(int t, int K, int T) {
		if (T < 1) {
			throw new IllegalArgumentException("The number of epochs must be postive!");
		}
		if (K < 1) {
			throw new IllegalArgumentException("The number of prototypes must be positive!");
		}
		if (t == T - 1) {
			return 0.01;
		}
		final double lambda_0 = (double) K / 2;
		if (t == 0) {
			return lambda_0;
		}
		return lambda_0 * Math.pow(0.01 / lambda_0, (double) t / (T - 1));
	}

	/**
	 * <p>
	 * Returns the strict assignments of all data points handled in the given
	 * Relational Neural Gas model to prototypes. The assignments are computed
	 * by the minimum distance to prototypes. More formally, it returns a vector assignments
	 * where</p>
	 *
	 * <p>
	 * assignment[i] := argmin_k d(w<sub>k</sub> , x<sub>i</sub>)
	 * </p>
	 *
	 * @param model an RNGModel.
	 *
	 * @return the strict assignments of all data points handled in the given
	 * Relational Neural Gas model to prototypes.
	 */
	public static int[] getAssignments(RNGModel model) {
		final int m = model.getNumberOfDatapoints();
		final int[] assignments = new int[m];
		final double[][] Dp = model.getDistancesToPrototypes();
		for (int i = 0; i < m; i++) {
			assignments[i] = ArrayFunctions.getMinIdx(Dp[i]);
		}
		return assignments;
	}

	/**
	 * Classifies new data according to a given RNGModel. In particular, assume that n new data
	 * points are given. Then, this method returns an n-dimensional array, where the j-th entry
	 * is the index of the cluster the j-th data point has been assigned to.
	 *
	 * @param D an n x m dimensional distance matrix containing the distances of the new data
	 * points to the training data points.
	 * @param model model a RNGModel.
	 * @return an n-dimensional array, where the j-th entry is the index of the cluster the j-th
	 * data point has been assigned to.
	 */
	public static int[] classify(double[][] D, RNGModel model) {
		final int n = D.length;
		final int[] classification = new int[n];
		for (int j = 0; j < n; j++) {
			classification[j] = classify(D[j], model);
		}
		return classification;
	}

	/**
	 * Classifies a new data point according to a given RNGModel. In particular,
	 * this method returns the index of the cluster the new data point has been assigned to.
	 *
	 * @param d an m x 1 dimensional vector containing the distances of the new data point to all
	 * training data points.
	 * @param model a RNGModel.
	 *
	 * @return an n-dimensional array, where the j-th entry is the index of the cluster the j-th
	 * data point has been assigned to.
	 */
	public static int classify(double[] d, RNGModel model) {
		final double[] dp = RelationalDistances.getDistancesToPrototypes(d, model);
		return ArrayFunctions.getMinIdx(dp);
	}

	/**
	 * <p>
	 * Returns an array with K entries, where each entry is another array containing the indices of
	 * all data points that have been assigned to the respective cluster/prototype. That is,
	 * entry k of the output array contains the set:</p>
	 *
	 * <p>{ i | getAssignments(model)[i] == k}</p>
	 *
	 * <p>The returned collection lists the data point indices in ascending order.</p>
	 *
	 * @param model a RNGModel.
	 *
	 * @return an array with K entries, where each entry is another array containing the indices of
	 * all data points that have been assigned to the respective cluster/prototype.
	 */
	public static int[][] getClusterMembers(RNGModel model) {
		final int K = model.getNumberOfPrototypes();
		final int[][] members = new int[K][];
		for (int k = 0; k < K; k++) {
			members[k] = getClusterMembers(model, k);
		}
		return members;
	}

	/**
	 * <p>
	 * Returns the indices of all data points that have been assigned to the
	 * cluster/prototype with index k. This is equivalent to the set</p>
	 *
	 * <p>{ i | getAssignments(model)[i] == k}</p>
	 *
	 * <p>The returned collection lists the data point indices in ascending order.</p>
	 *
	 * @param model a RNGModel.
	 * @param k the index of a cluster/prototype.
	 *
	 * @return the indices of all data points that have been assigned to the
	 * cluster with index k.
	 */
	public static int[] getClusterMembers(RNGModel model, int k) {
		return getClusterMembers(model, k, getAssignments(model));
	}

	/**
	 * <p>
	 * Returns the indices of all data points that have been assigned to the
	 * cluster/prototype with index k. This is equivalent to the set</p>
	 *
	 * <p>{ i | getAssignments(model)[i] == k}</p>
	 *
	 * <p>The returned collection lists the data point indices in ascending order.</p>
	 *
	 * @param model a RNGModel.
	 * @param k the index of a cluster/prototype.
	 * @param assignments the assignments vector for the given model (result of
	 * "getAssignments").
	 *
	 * @return the indices of all data points that have been assigned to the
	 * cluster with index k.
	 */
	public static int[] getClusterMembers(RNGModel model, int k, int[] assignments) {
		final ArrayList<Integer> members = new ArrayList<>();
		for (int i = 0; i < assignments.length; i++) {
			if (assignments[i] == k) {
				members.add(i);
			}
		}
		final int[] memberArr = new int[members.size()];
		for (int i = 0; i < members.size(); i++) {
			memberArr[i] = members.get(i);
		}

		return memberArr;
	}

	/**
	 * <p>Returns an array with K entries, where entry k is the index of the data point which is
	 * closest to the relational prototype for cluster k. More formally, entry k is:</p>
	 *
	 * <p>
	 * argmin_i d(w<sub>k</sub>, x<sub>i</sub>)
	 * </p>
	 *
	 * @param model a RNGModel.
	 *
	 * @return an array with K entries, where entry k is the index of the data point which is
	 * closest to the relational prototype for cluster k.
	 */
	public static int[] getExamplars(RNGModel model) {
		final int m = model.getNumberOfDatapoints();
		final int K = model.getNumberOfPrototypes();
		final double[][] Dp = model.getDistancesToPrototypes();
		final int[] exemplars = new int[K];
		for (int k = 0; k < K; k++) {
			for (int i = 1; i < m; i++) {
				if (Dp[i][k] < Dp[exemplars[k]][k]) {
					exemplars[k] = i;
				}
			}
		}
		return exemplars;
	}

}
