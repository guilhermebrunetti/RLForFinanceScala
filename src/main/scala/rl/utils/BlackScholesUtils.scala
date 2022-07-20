package rl.utils

import breeze.numerics._
import breeze.stats.distributions


object BlackScholesUtils {
  
  def europeanOptionPrice(
    spotPrice: Double,
    strike: Double,
    expiry: Double,
    vol: Double,
    rate: Double = 0.0,
    isCall: Boolean = true,
  ): Double = {
    val sign = if (isCall) 1.0 else -1.0
    if (expiry > 0) {
      val totalVariance = vol * vol * expiry
      val sqrtTV = sqrt(totalVariance)
      val df = exp(-rate * expiry)
      val forwardPrice = spotPrice / df
      val logMoneyness = log(forwardPrice / strike)
      val d1 = logMoneyness / sqrtTV + 0.5 * sqrtTV
      val d2 = d1 - sqrtTV
      val dist = distributions.Gaussian(0.0, 1.0)
      df * sign * (forwardPrice * dist.cdf(sign * d1) - strike * dist.cdf(sign * d2))
    }
    else
      math.max(sign * (spotPrice - strike), 0.0)
  }
  
  def optionPayoff(
    spotPrice: Double,
    strike: Double,
    isCall: Boolean = true
  ): Double = {
    val sign = if (isCall) 1.0 else -1.0
    math.max(sign * (spotPrice - strike), 0.0)
  }
  
}
