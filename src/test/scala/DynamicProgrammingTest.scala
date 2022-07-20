import java.util.Locale

import breeze.numerics._
import org.scalactic._
import org.scalatest.FunSuite
import rl.FiniteMarkovRewardProcess.RewardTransition
import rl._
import rl.DynamicProgramming._
import rl.FiniteHorizon._
import rl.utils.Categorical

/**
 * A version of FlipFlop implemented with the FiniteMarkovProcess
 * machinery.
 */
class FlipFlop(val transitionRewardMap: RewardTransition[Boolean]) extends FiniteMarkovRewardProcess[Boolean] {}

class DynamicProgrammingTest extends FunSuite {
  
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)
  Locale.setDefault(Locale.US)
  
  val probability: Double = 0.7
  val finiteFlipFlop: FlipFlop = FlipFlop(probability)
  
  test("Evaluate MRP") {
    val gamma = 0.99
    val valueFunction: ValueFunction[Boolean] = evaluateMarkovRewardProcessResult(finiteFlipFlop, gamma)
    
    assert(valueFunction.size === 2)
    assert(valueFunction.values.map(x => abs(x - 170)).max < 0.1)
  }
  
  test("Compare to Backward Induction") {
    val finiteHorizonMRP = finiteHorizonMarkovRewardProcess(finiteFlipFlop, 10)
  
    val gamma = 1.0
    val valueFunction: ValueFunction[WithTime[Boolean]] = evaluateMarkovRewardProcessResult(finiteHorizonMRP, gamma)
    val finiteValueFunction: Seq[ValueFunction[Boolean]] = evaluate(unwrapFiniteHorizonMRP(finiteHorizonMRP), gamma)
  
    assert(valueFunction.size === 20)
  
    val trueState = NonTerminal(true)
    val falseState = NonTerminal(false)
  
    finiteValueFunction.zipWithIndex.foreach { case (vf, time) =>
      val trueStateWithTime = trueState.map(s => WithTime(s, time))
      val falseStateWithTime = falseState.map(s => WithTime(s, time))
      assert(valueFunction(trueStateWithTime) === vf(trueState))
      assert(valueFunction(falseStateWithTime) === vf(falseState))
    }
  }
  
}

object FlipFlop {
  
  def apply(probability: Double): FlipFlop =
    new FlipFlop(FiniteMarkovRewardProcess.processInputMap(inputMap(probability)))
  
  def inputMap(probability: Double): Map[Boolean, Categorical[(Boolean, Double)]] =
    Seq(true, false).map { b =>
      b -> Categorical(Map(
        (!b, 2.0) -> probability,
        (b, 1.0) -> (1.0 - probability)
      ))
    }.toMap
}
