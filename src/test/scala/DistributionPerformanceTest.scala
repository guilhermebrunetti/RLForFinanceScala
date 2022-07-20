import java.util.Locale

import breeze.stats.distributions
import com.typesafe.scalalogging.Logger
import org.scalameter._
import org.scalatest.FunSuite
import rl.utils

import scala.collection.parallel.CollectionConverters._
import scala.collection.parallel.immutable.ParVector

class DistributionPerformanceTest extends FunSuite {
  
  Locale.setDefault(Locale.US) // To print numbers in US format
  val logger: Logger = Logger("DistributionPerformanceTest")
  
  val numSamples: Int = 100000
  val poissonMean: Double = 1.0
  
  val mu: Double = 0.0
  val sigma: Double = 1.0
  
  val numRepeats: Int = 30
  
  test("Measure Performance of Poisson Random Number generation") {
    val avgTime1 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        distributions.Poisson(mean = poissonMean).sample(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
    
    val avgTime2 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        utils.Poisson(mean = poissonMean).samples(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
    
    val avgTime3 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        utils.Poisson(mean = poissonMean).samplesPar(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
    
    logger.info(f"Generating Poisson random variables")
    logger.info(f"First run (from Breeze): $avgTime1%1.4f")
    logger.info(f"Second run (from rl.Utils): $avgTime2%1.4f")
    logger.info(f"Third run (from rl.Utils in parallel): $avgTime3%1.4f")
  }
  
  test("Measure Performance of Gaussian Random Number generation") {
    val avgTime1 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        distributions.Gaussian(mu = mu, sigma = sigma).sample(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
  
    val avgTime2 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        utils.Gaussian(mu = mu, sigma = sigma).samples(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
  
    val avgTime3 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        utils.Gaussian(mu = mu, sigma = sigma).samplesPar(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
  
    logger.info(f"Generating Gaussian random variables")
    logger.info(f"First run (from Breeze): $avgTime1%1.4f")
    logger.info(f"Second run (from rl.Utils): $avgTime2%1.4f")
    logger.info(f"Third run (from rl.Utils in parallel): $avgTime3%1.4f")
  }
  
  test("Breeze with parallel collections"){
    val dist = distributions.Gaussian(mu = mu, sigma = sigma)
  
    val avgTime1 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        dist.sample(numSamples)
      }
      x.value
    }.sum / numRepeats.toDouble
  
    val avgTime2 = (0 until numRepeats).par.map { _ =>
      val x: Quantity[Double] = withWarmer(new Warmer.Default) measure {
        ParVector.fill(numSamples)(dist.sample())
      }
      x.value
    }.sum / numRepeats.toDouble
  
    logger.info(f"Breeze with parallel collection")
    logger.info(f"First run (vanilla Breeze): $avgTime1%1.4f")
    logger.info(f"Second run (Breeze with parallel collection): $avgTime2%1.4f")
  }
  
}
