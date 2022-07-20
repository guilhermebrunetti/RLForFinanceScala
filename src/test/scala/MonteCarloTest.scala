import java.util.Locale

import breeze.numerics._
import org.scalactic._
import org.scalatest.FunSuite
import rl.ApproximateDynamicProgramming.{ValueFunctionApproximation, almostEqual}
import rl._
import rl.utils.Choose

class MonteCarloTest extends FunSuite {
  
  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)
  Locale.setDefault(Locale.US)
  
  val probability: Double = 0.7
  val finiteFlipFlop: FlipFlop = FlipFlop(probability)
  
  test("Evaluate Finite MRP") {
    val gamma = 0.99
    
    val initialValueMap = Map(NonTerminal(true) -> 0.0, NonTerminal(false) -> 0.0)
    val initialFunctionApproximation: Tabular[NonTerminal[Boolean]] = Tabular(valuesMap = initialValueMap)
    
    val traces: LazyList[LazyList[TransitionStep[Boolean]]] = finiteFlipFlop.rewardTraces(
      Choose(
        Seq(NonTerminal(true), NonTerminal(false))
      )
    )
    
    val valueFunction = IterateUtils.converged[ValueFunctionApproximation[Boolean]](
      MonteCarlo.mcPrediction(traces, initialFunctionApproximation, gamma),
      (x, y) => almostEqual(x, y)(tolerance = 0.001)
    )
  
    val vfMap = valueFunction.asInstanceOf[Tabular[NonTerminal[Boolean]]].valuesMap
    
    assert(vfMap.size === 2)
    assert(vfMap.values.map(x => abs(x - 170)).max < 1.0)
  }
  
  test("Test Returns calculation from Transition Step"){
    
    val gamma = 0.1
    
    val transitionSteps: Seq[TransitionStepMRP[Int]] = Seq.range(0, 10).map{t =>
      TransitionStepMRP(NonTerminal(t), NonTerminal(t + 1), (t + 1) % 2)
    }
    
    val returnSteps = Returns.returns(transitionSteps, gamma = gamma)
    
    assert(returnSteps.head.returns === 1.01010101)
  }
  
}
