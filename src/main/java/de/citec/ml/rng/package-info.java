/*
 * Relational Neural Gas
 
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
/**
 * This is an implementation of the Neural Gas algorithm on
 * distance data (Relational Neural Gas) for unsupervised clustering.
 *
 * We recommend that you use the functions provided by the RelationalNeuralGas
 * class for your purposes. All other classes and functions are utilities which
 * are used by this central class. In particular, you can use RelationalNeuralGas.train()
 * to obtain a RNGModel (i.e. a clustering of your data), and subsequently
 * you can use RelationalNeuralGas.getAssignments() to obtain the resulting
 * cluster assignments, and RelationalNeuralGas.classify() to cluster new points
 * which are not part of the training data set.
 *
 * The underlying scientific work is summarized nicely in the dissertation
 * "Topographic Mapping of Dissimilarity Datasets" by Alexander Hasenfuss
 * (2009).
 *
 * The basic properties of an Relational Neural Gas algorithm are the following:
 * 1.) It is relational: The data is represented only in terms of a pairwise
 * distance matrix.
 * 2.) It is a clustering method: The algorithm provides a clustering model,
 * that is: After calculation,
 * each data point should be assigned to a cluster (for this package here we
 * only consider hard clustering, that is: each data point is assigned to
 * exactly one cluster).
 * 3.) It is a vector quantization method: Each cluster corresponds to a
 * prototype, which is in the center of the
 * cluster and data points are assigned to the cluster if and only if they are
 * closest to this particular prototype.
 * 4.) It is rank-based: The updates of the prototypes depend only on
 * the distance ranking, not on the absolute value of the distances.
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
package de.citec.ml.rng;
