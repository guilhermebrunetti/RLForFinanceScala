import java.util.Locale

import breeze.numerics._
import org.scalactic._
import org.scalatest.FunSuite
import rl.utils.{Constant, FiniteDistribution, Poisson}

class DistributionTest extends FunSuite {
  
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)
  Locale.setDefault(Locale.US)
  
  val testValue: Double = 1.0
  val numSamples: Int = 10
  val probabilityTable: Map[Int, Double] = Map(1 -> 0.5, 2 -> 0.25, 4 -> 0.25)
  val cumulativeProbability: Seq[(Int, Double)] = Seq(1 -> 0.5, 2 -> 0.75, 4 -> 1.0)
  val constant: Constant[Double] = Constant(testValue)
  val finiteDistribution: FiniteDistribution[Int] = FiniteDistribution(probabilityTable)
  val poissonMean: Double = 1.0
  val poissonDistribution: Poisson = Poisson(poissonMean)

  def testFunction(x: Double): Double = x * x
  
  test("Constant Distribution should return constant value") {
    assert(constant.sample === testValue)
    assert(constant.expectation(testFunction) === testFunction(testValue))
    assert(constant.map(testFunction).sample === testFunction(testValue))
    assert(constant.samples(numSamples).toSet.size === 1)
  }
  
  test("Distribution samples should return a sequence of size numSamples") {
    assert(constant.samples(numSamples).length === numSamples)
  }
  
  test("Cumulative probability test for FiniteDistribution") {
    assert(finiteDistribution.cumulativeProbabilitySeq === cumulativeProbability)
  }
  
  test("Test for FiniteDistribution.sampleOutcome method") {
    assert(finiteDistribution.sampleOutcome(-0.1) === 1)
    assert(finiteDistribution.sampleOutcome(0.00) === 1)
    assert(finiteDistribution.sampleOutcome(0.25) === 1)
    assert(finiteDistribution.sampleOutcome(0.50) === 2)
    assert(finiteDistribution.sampleOutcome(0.60) === 2)
    assert(finiteDistribution.sampleOutcome(0.75) === 4)
    assert(finiteDistribution.sampleOutcome(0.80) === 4)
    assert(finiteDistribution.sampleOutcome(1.00) === 4)
    assert(finiteDistribution.sampleOutcome(1.10) === 4)
  }
  
  test("Test for FiniteDistribution.expectation method") {
    assert(finiteDistribution.expectation(x => x.toDouble) === 2.0)
    assert(finiteDistribution.expectation(x => testFunction(x)) === 5.5)
  }
  
  test("Test for FiniteDistribution.map method") {
    assert(
      finiteDistribution
        .map(x => x % 2)
        .probabilityTable === Map(0 -> 0.5, 1 -> 0.5))
  }
  
  test("Test for Poisson distribution") {
    def fact(n: Int): Int = if (n > 0) (1 to n).product else 1
    
    def f(n: Int): Double = exp(-poissonMean) * pow(poissonMean, -n) / fact(n)
    
    def s(n: Int): Double = (0 to n).map(f).sum
    
    def probabilities(n: Int): Double = (0 to n).map { i =>
      if (i < n)
        poissonDistribution.probabilityMassFunction(i)
      else
        1.0 - poissonDistribution.cumulativeDistributionFunction(n - 1)
    }.sum
    
    assert(poissonDistribution.probabilityMassFunction(-1) === 0.0)
    assert(poissonDistribution.probabilityMassFunction(0) === f(0))
    assert(poissonDistribution.probabilityMassFunction(1) === f(1))
    assert(poissonDistribution.probabilityMassFunction(2) === f(2))
    assert(poissonDistribution.probabilityMassFunction(3) === f(3))
    assert(poissonDistribution.cumulativeDistributionFunction(-1) === 0.0)
    assert(poissonDistribution.cumulativeDistributionFunction(0) === s(0))
    assert(poissonDistribution.cumulativeDistributionFunction(1) === s(1))
    assert(poissonDistribution.cumulativeDistributionFunction(2) === s(2))
    assert(poissonDistribution.cumulativeDistributionFunction(3) === s(3))
    assert(poissonDistribution.quantile(0.0) === 0)
    assert(poissonDistribution.quantile(0.30) === 0)
    assert(poissonDistribution.quantile(0.73) === 1)
    assert(poissonDistribution.quantile(0.90) === 2)
    assert(poissonDistribution.quantile(0.98) === 3)
    assert(poissonDistribution.quantile(0.99) === 4)
    assert(probabilities(1) === 1.0)
    assert(probabilities(2) === 1.0)
    assert(probabilities(3) === 1.0)
    assert(probabilities(4) === 1.0)
    assert(probabilities(5) === 1.0)
  }
  
}
