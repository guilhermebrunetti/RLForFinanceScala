import java.util.Locale

import breeze.numerics._
import org.scalactic._
import org.scalatest.FunSuite
import rl.ApproximateDynamicProgramming._
import rl.DynamicProgramming.ValueFunction
import rl.FiniteHorizon._
import rl._
import rl.utils.Choose

class ApproximateDynamicProgrammingTest extends FunSuite {
  
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)
  Locale.setDefault(Locale.US)
  
  val probability: Double = 0.7
  val finiteFlipFlop: FlipFlop = FlipFlop(probability)
  implicit val tolerance: Double = 1.0e-4
  
  test("Test evaluate finite MRP") {
    
    val initialValues: Map[NonTerminal[Boolean], Double] = finiteFlipFlop.nonTerminalStates.map(k => k -> 0.0).toMap
    val initialFunctionApproximation: Dynamic[NonTerminal[Boolean]] = Dynamic(initialValues)
    
    val gamma = 0.99
    
    val valueFunction = evaluateFiniteMRPResult(
      mrp = finiteFlipFlop,
      gamma = gamma,
      initialApproximation = initialFunctionApproximation
    )(tolerance)
    
    val vfMap = valueFunction.asInstanceOf[Dynamic[NonTerminal[Boolean]]].valuesMap
    
    assert(vfMap.size === 2)
    assert(vfMap.values.map(x => abs(x - 170)).max < 0.1)
  }
  
  test("Test evaluate MRP") {
    
    val initialValues: Map[NonTerminal[Boolean], Double] = finiteFlipFlop.nonTerminalStates.map(k => k -> 0.0).toMap
    val initialFunctionApproximation: Dynamic[NonTerminal[Boolean]] = Dynamic(initialValues)
    
    val gamma = 0.99
    
    val ntDistribution = Choose(finiteFlipFlop.nonTerminalStates)
    val numSamples = 30
    
    val valueFunction = evaluateMarkovRewardProcessResult(
      mrp = finiteFlipFlop,
      gamma = gamma,
      initialApproximation = initialFunctionApproximation,
      ntStateDistribution = ntDistribution,
      numSamples = numSamples
    )(tolerance)
    
    val vfFinite = evaluateFiniteMRPResult(
      mrp = finiteFlipFlop,
      gamma = gamma,
      initialApproximation = initialFunctionApproximation
    )(tolerance)
    
    assert(almostEqual(valueFunction, vfFinite)(tolerance = 0.1))
  }
  
  test("Test Backward Induction") {
    
    val finiteHorizonMRP: FiniteMarkovRewardProcess[WithTime[Boolean]] =
      finiteHorizonMarkovRewardProcess(finiteFlipFlop, 10)
  
    val initialValues: Map[NonTerminal[WithTime[Boolean]], Double] =
      finiteHorizonMRP.nonTerminalStates.map(k => k -> 0.0).toMap
    
    val initialFunctionApproximation: Dynamic[NonTerminal[WithTime[Boolean]]] = Dynamic(initialValues)
    
    val gamma = 1.0
    
    val vfFinite: ValueFunctionApproximation[WithTime[Boolean]] = evaluateFiniteMRPResult(
      mrp = finiteHorizonMRP,
      gamma = gamma,
      initialApproximation = initialFunctionApproximation
    )(tolerance)
  
    val finiteValueFunction: Seq[ValueFunction[Boolean]] = evaluate(unwrapFiniteHorizonMRP(finiteHorizonMRP), gamma)
  
    val vfMap = vfFinite.asInstanceOf[Dynamic[NonTerminal[Boolean]]].valuesMap
  
    assert(vfMap.size === 20)
  
    val trueState = NonTerminal(true)
    val falseState = NonTerminal(false)
  
    finiteValueFunction.zipWithIndex.foreach { case (vf, time) =>
      val trueStateWithTime: NonTerminal[WithTime[Boolean]] = trueState.map(s => WithTime(s, time))
      val falseStateWithTime: NonTerminal[WithTime[Boolean]] = falseState.map(s => WithTime(s, time))
      assert(vfFinite(trueStateWithTime) === vf(trueState))
      assert(vfFinite(falseStateWithTime) === vf(falseState))
    }
    
  }
  
}
