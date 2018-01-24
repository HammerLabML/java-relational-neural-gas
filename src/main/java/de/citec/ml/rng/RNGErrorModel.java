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
 * An RNGModel that additionally implements this interface also provides
 * information on the quantization error at each training epoch.
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public interface RNGErrorModel extends RNGModel {

	/**
	 * Returns the number of training epochs for this model.
	 *
	 * @return the number of training epochs for this model.
	 */
	public int getNumberOfEpochs();

	/**
	 * Returns the quantization errors in each epoch.
	 *
	 * @return the quantization errors in each epoch.
	 */
	public double[] getQuantizationErrors();
}
