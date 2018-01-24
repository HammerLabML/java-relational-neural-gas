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
 * This is a collections of functions on primitive double arrays.
 *
 * @author Benjamin Paassen - bpaassen(at)techfak.uni-bielefeld.de
 */
public final class ArrayFunctions {

	private ArrayFunctions() {

	}

	/**
	 * Returns the index of the minimum element in the given array.
	 *
	 * @param d a double array.
	 *
	 * @return the index of the minimum element in the given array.
	 */
	public static int getMinIdx(double[] d) {
		if (d.length == 0) {
			throw new IllegalArgumentException("The input array is empty!");
		}
		int minIdx = 0;
		double min = d[0];
		for (int i = 1; i < d.length; i++) {
			if (d[i] < min) {
				minIdx = i;
				min = d[minIdx];
			}
		}
		return minIdx;
	}

	/**
	 * Returns the index of the maximum element in the given array.
	 *
	 * @param d a double array.
	 *
	 * @return the index of the maximum element in the given array.
	 */
	public static int getMaxIdx(double[] d) {
		if (d.length == 0) {
			throw new IllegalArgumentException("The input array is empty!");
		}
		int maxIdx = 0;
		double max = d[0];
		for (int i = 1; i < d.length; i++) {
			if (d[i] > max) {
				maxIdx = i;
				max = d[maxIdx];
			}
		}
		return maxIdx;
	}

}
