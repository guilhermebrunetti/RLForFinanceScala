package rl.utils

import breeze.numerics._

object Utils extends App {
  
  val VSML: Double = 1.0e-8
  
  def getLogisticFunction(alpha: Double)(x: Double): Double = {
    1.0 / (1.0 + exp(-alpha * x))
  }
  
  def getUnitSigmoidFunction(alpha: Double)(x: Double): Double = {
    1.0 / (1.0 + pow(1.0 / (if (x == 0) VSML else x) - 1.0, alpha))
  }
  
  def laguerrePolynomials: Seq[Double => Double] = Seq(
    _ => 1.0,
    x => 1.0 - x,
    x => (pow(x, 2) - 4 * x + 2.0) / 2.0,
    x => (-pow(x, 3) + 9 * pow(x, 2) - 18 * x + 6) / 6.0,
    x => (pow(x, 4) - 16 * pow(x, 3) + 72 * pow(x, 2) - 96 * x + 24) / 24.0,
    x => (-pow(x, 5) + 25 * pow(x, 4) - 200 * pow(x, 3) + 600 * pow(x, 2) - 600 * x + 120) / 120.0
  )
  
  def hermitePolynomials: Seq[Double => Double] = Seq(
    _ => 1.0,
    x => x,
    x => pow(x, 2) - 1.0,
    x => pow(x, 3) - 3 * x,
    x => pow(x, 4) - 6 * pow(x, 2) + 3,
    x => pow(x, 5) - 10 * pow(x, 3) + 15 * x
  )
  
}
