package rl.utils

import breeze.stats._
import com.typesafe.scalalogging.Logger
import rl.utils.Categorical.normalizeMassTable
import rl.utils.Choose.massTable
import rl.utils.SampledDistribution.defaultExpectationSamples

import scala.collection.parallel.CollectionConverters._
import scala.collection.parallel.immutable.ParSeq

trait Distribution[A] {
  /**
   * Returns a random sample from distribution
   */
  def sample: A
  
  /**
   * Returns N random samples from distribution
   */
  def samples(n: Int): Seq[A] = {
    (0 until n).map(_ => sample)
  }
  
  /**
   * Returns N random samples from distribution (Parallel Collection)
   */
  def samplesPar(n: Int): ParSeq[A] = {
    (0 until n).par.map(_ => sample)
  }
  
  /**
   * Returns the expectation of f(X)
   */
  def expectation(f: A => Double): Double
  
  def expectationPar(f: A => Double): Double = expectation(f)
  
  def map[B](f: A => B): Distribution[B] = {
    new SampledDistribution(() => f(this.sample))
  }
  
  def flatMap[B](f: A => Distribution[B]): Distribution[B] = {
    new SampledDistribution(() => f(this.sample).sample)
  }
}

class SampledDistribution[A](
  val sampler: () => A,
  val expectationSamples: Int = defaultExpectationSamples
) extends Distribution[A] {
  /**
   * A distribution defined by a function to sample it
   */
  override def sample: A = sampler()
  
  override def expectation(f: A => Double): Double = {
    val N: Double = expectationSamples.toDouble
    samples(expectationSamples).map(x => f(x) / N).sum
  }
  
  override def expectationPar(f: A => Double): Double = {
    val N: Double = expectationSamples.toDouble
    samplesPar(expectationSamples).par.map(x => f(x) / N).sum
  }
}

object SampledDistribution {
  
  val defaultExpectationSamples: Int = 10000
  
  def apply[A](
    sampler: () => A,
    expectationSamples: Int = defaultExpectationSamples
  ): SampledDistribution[A] = new SampledDistribution(sampler, expectationSamples)
}

class Uniform(
  val low: Double = 0.0, // Lower boundary
  val high: Double = 1.0, // Upper boundary
  override val expectationSamples: Int = defaultExpectationSamples
) extends SampledDistribution(
  sampler = () => distributions.Uniform(low, high).draw(),
  expectationSamples = expectationSamples
) {}

class Bernoulli(
  val p: Double = 0.5,
  override val expectationSamples: Int = defaultExpectationSamples
) extends SampledDistribution(
  sampler = () => distributions.Bernoulli(p = p).draw(),
  expectationSamples = expectationSamples
) {
  override def expectation(f: Boolean => Double): Double = p * f(true) + (1 - p) * f(false)
}

object Bernoulli {
  def apply(
    p: Double,
    expectationSamples: Int = defaultExpectationSamples
  ): Bernoulli = new Bernoulli(p, expectationSamples)
}

class Poisson(
  val mean: Double = 1.0, // Mean of distribution
  override val expectationSamples: Int = defaultExpectationSamples
) extends SampledDistribution(
  sampler = () => distributions.Poisson(mean = mean).draw(),
  expectationSamples = expectationSamples
) {
  
  def probabilityMassFunction(x: Int): Double = {
    if (x < 0) 0.0 else distributions.Poisson(mean = mean).probabilityOf(x)
  }
  
  def cumulativeDistributionFunction(x: Int): Double = {
    if (x < 0) 0.0 else distributions.Poisson(mean = mean).cdf(x)
  }
  
  def quantile(probability: Double): Int = {
    require(probability >= 0.0, s"Probability should be positive number, got $probability instead")
    require(probability < 1.0, s"Probability should be strictly less than 1.0, got $probability instead")
    val dist = distributions.Poisson(mean = mean)
    LazyList.from(0).dropWhile(i => dist.cdf(i) < probability).head
  }
}

object Poisson {
  
  def apply(
    mean: Double,
    expectationSamples: Int = defaultExpectationSamples
  ): Poisson = new Poisson(mean, expectationSamples)
  
}

class Gaussian(
  val mu: Double = 0.0, // Mean of distribution
  val sigma: Double = 1.0, // Standard Deviation of distribution
  override val expectationSamples: Int = defaultExpectationSamples
) extends SampledDistribution(
  sampler = () => distributions.Gaussian(mu = mu, sigma = sigma).draw(),
  expectationSamples = expectationSamples
) {}

object Gaussian {
  def apply(
    mu: Double = 0.0,
    sigma: Double = 1.0,
    expectationSamples: Int = defaultExpectationSamples
  ): Gaussian = new Gaussian(mu, sigma, expectationSamples)
}

class LogGaussian(
  val mu: Double = 0.0, // Mean of distribution
  val sigma: Double = 1.0, // Standard Deviation of distribution
  override val expectationSamples: Int = defaultExpectationSamples
) extends SampledDistribution(
  sampler = () => distributions.LogNormal(mu = mu, sigma = sigma).draw(),
  expectationSamples = expectationSamples
) {}

object LogGaussian {
  def apply(
    mu: Double = 0.0,
    sigma: Double = 1.0,
    expectationSamples: Int = defaultExpectationSamples
  ): LogGaussian = new LogGaussian(mu, sigma, expectationSamples)
}

class FiniteDistribution[A](val probabilityTable: Map[A, Double])
  extends Distribution[A] {
  require(probabilityTable.nonEmpty, s"ProbabilityTable cannot be empty")
  require(probabilityTable.values.sum <= 1.01 && probabilityTable.values.sum >= 0.99,
    s"Sum of probabilities need to be 1.0. instead, got ${probabilityTable.values.sum}"
  )
  
  val cumulativeProbabilitySeq: Seq[(A, Double)] = {
    val (outcomes, probabilities) = probabilityTable.toSeq.unzip
    val cumulativeProbabilities = probabilities.scanLeft(0.0)(_ + _).tail
    outcomes.zip(cumulativeProbabilities)
  }
  
  private val lastElement = cumulativeProbabilitySeq.last
  
  override def sample: A = {
    val randomDraw = distributions.Rand.uniform.draw()
    sampleOutcome(randomDraw)
  }
  
  def sampleOutcome(probability: Double): A = {
    val (outcome, _) = cumulativeProbabilitySeq
      .dropWhile { case (_, cp) => cp <= probability }
      .headOption
      .getOrElse(lastElement)
    outcome
  }
  
  def probability(outcome: A): Double = {
    probabilityTable.getOrElse(outcome, 0.0)
  }
  
  override def map[B](f: A => B): FiniteDistribution[B] = {
    val baseMap: Map[A, Double] = this.probabilityTable
    val newMap: Map[B, Double] = baseMap
      .groupMapReduce { case (a, _) => f(a) } { case (_, p) => p } {
        _ + _
      }
    
    new FiniteDistribution(newMap)
  }
  
  def flatMap[B](f: A => FiniteDistribution[B]): FiniteDistribution[B] = {
    val baseMap: Map[A, Double] = this.probabilityTable
    val auxiliaryMap: Map[(A, B), Double] = baseMap.flatMap { case (a, p1) =>
      f(a).probabilityTable.map { case (b, p2) => (a, b) -> (p1 * p2) }
    }
    val newMap: Map[B, Double] = auxiliaryMap
      .groupMapReduce { case ((_, b), _) => b } { case (_, p) => p } {
        _ + _
      }
    
    new FiniteDistribution(newMap)
  }
  
  override def expectation(f: A => Double): Double = {
    probabilityTable.map { case (a, p) => f(a) * p }.sum
  }
  
  override def toString: String = {
    val tableToString = probabilityTable
      .view
      .mapValues(p => f"$p%1.4f")
      .toMap
      .mkString("\n\t")
    s"FiniteDistribution(\n\t$tableToString\n\t)"
  }
}

object FiniteDistribution {
  def apply[A](probabilityTable: Map[A, Double]): FiniteDistribution[A] =
    new FiniteDistribution(probabilityTable)
}

class Constant[A](val value: A)
  extends FiniteDistribution[A](Map(value -> 1.0)) {
  
  override def sample: A = value
  
  override def expectation(f: A => Double): Double = f(value)
  
  override def map[B](f: A => B): Constant[B] = {
    new Constant(f(this.value))
  }
}

object Constant {
  def apply[A](value: A): Constant[A] = new Constant(value)
}

class Categorical[A](val massTable: Map[A, Double])
  extends FiniteDistribution[A](normalizeMassTable(massTable)) {
  require(massTable.values.forall(_ >= 0), s"All weights should be positive, got negative values")
}

object Categorical {
  
  def normalizeMassTable[A](massTable: Map[A, Double]): Map[A, Double] = {
    val total = massTable.values.sum
    massTable.view.mapValues(x => x / total).toMap
  }
  
  def fromIterable[A](elements: Iterable[(A, Double)]): Categorical[A] = {
    val massTable: Map[A, Double] = elements
      .groupMapReduce { case (k, _) => k } { case (_, v) => v }(_ + _)
    Categorical(massTable)
  }
  
  def apply[A](massTable: Map[A, Double]): Categorical[A] = new Categorical(massTable)
}

class Choose[A](val elements: Iterable[A])
  extends Categorical[A](massTable(elements)) {
  private val elementSeq: IndexedSeq[A] = elements.toIndexedSeq
  
  override def sample: A = {
    
    val randomDraw = distributions.Rand.randInt(elementSeq.length).draw()
    elementSeq(randomDraw)
  }
}

object Choose {
  
  def massTable[A](elements: Iterable[A]): Map[A, Double] = elements.map(e => e -> 1.0).toMap
  
  def apply[A](elements: Iterable[A]): Choose[A] = new Choose(elements)
}

object DistributionApp extends App {
  
  val logger: Logger = Logger("DistributionApp")
  
  val probabilityTable = Map(1 -> 0.5, 2 -> 0.25, 3 -> 0.25)
  val numSamples = 1000
  
  val fd = new FiniteDistribution(probabilityTable)
  
  val samples = fd.samples(numSamples)
  
  logger.info(s"FiniteDistribution sample (first 20):\n${samples.take(20)}")
  
  val mapReduce = samples
    .groupMapReduce(x => x)(_ => (1.0 / numSamples).toFloat)(_ + _)
  
  logger.info(s"Theoretical Distribution:\n$probabilityTable")
  logger.info(s"Empirical Distribution:\n$mapReduce")
  
}
