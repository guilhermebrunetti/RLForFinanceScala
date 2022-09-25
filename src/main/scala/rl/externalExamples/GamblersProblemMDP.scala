package rl.externalExamples

import breeze.plot.{Figure, plot}
import com.typesafe.scalalogging.Logger
import rl.DynamicProgramming._
import rl.FiniteMarkovDecisionProcess.{StateActionMapping, processInputMap}
import rl.utils.Categorical
import rl.{FiniteDeterministicPolicy, FiniteMarkovDecisionProcess, NonTerminal}

import java.time.LocalDateTime
import java.util.Locale

/** *
 * From Sutton and Barto's RL Book, 2nd Edition, Example 4.3:
 * "
 * A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
 * If the coin comes up heads, he wins as many dollars as he has staked on that flip;
 * if it is tails, he loses his stake.
 * The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money.
 * On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars.
 *
 * The state-value function then gives the probability of winning from each state.
 * A policy is a mapping from levels of capital to stakes.
 * The optimal policy maximizes the probability of reaching the goal.
 * "
 *
 * For this problem, the optimal policy is not unique and in our algorithm it will depend on
 * the tie-breaking rule for the argmax selection
 */
class GamblersProblemMDP(
  val goal: Int,
  val probability: Double,
) extends FiniteMarkovDecisionProcess[Int, Int] {
  require(
    probability > 0.0 && probability < 1.0,
    s"Probability should be a number strictly between 0 and 1; got instead $probability"
  )

  require(goal > 0, s"Goal should be a strictly positive integer numbers; got instead $goal")

  override val stateActionMap: StateActionMapping[Int, Int] = {
    val inputMap = (1 until goal).map { i =>
      val maxBet = math.min(i, goal - i)
      i -> (1 to maxBet).map { a =>
        a -> Categorical(
          Map(
            (i + a, if (i + a == goal) 1.0 else 0.0) -> probability,
            (i - a, 0.0) -> (1.0 - probability)
          )
        )
      }.toMap
    }.toMap

    processInputMap(inputMap)
  }

  override def stateSortingFunction(x: NonTerminal[Int], y: NonTerminal[Int]): Boolean = x.state <= y.state

  /**
   *
   * The actionSortingFunction will be used to resolve ties in the arg max during value iteration.
   * Because there is no unique optimal policy for this problem, we will get different optimal policies
   * depending on the choice of sorting criteria.
   * For x <= y, we get max-bet policy
   * For x > y, we get the graph from Sutton and Barto's book.
   */
  override def actionSortingFunction(x: Int, y: Int): Boolean = x > y
}

object GamblersProblemMDP {
  def apply(goal: Int, probability: Double): GamblersProblemMDP =
    new GamblersProblemMDP(goal, probability)

}

object GamblersProblemMDPApp extends App {

  val logger: Logger = Logger("GamblersProblemMDPApp")
  Locale.setDefault(Locale.US) // To print numbers in US format

  val goal: Int = 100
  val probability: Double = 0.40
  val gamblersProblemMDP = GamblersProblemMDP(goal, probability)
  val sortingFunction: (NonTerminal[Int], NonTerminal[Int]) => Boolean = gamblersProblemMDP.stateSortingFunction

  val gamma = 1.0

  val nonTerminalStates = gamblersProblemMDP.nonTerminalStates

  logger.info(s"Starting computation at ${LocalDateTime.now()}")
  val (optimalValueFunction: ValueFunction[Int], optimalPolicy: FiniteDeterministicPolicy[Int, Int]) =
    valueIterationResult(gamblersProblemMDP, gamma)
  logger.info(s"Finished computation at ${LocalDateTime.now()}")

  // Policy of making of always betting all the money the agent has
  val actionMap = (1 until goal).map(i => i -> math.min(i, goal - i)).toMap
  val maxBetPolicy = FiniteDeterministicPolicy(actionMap = actionMap)

  val impliedMRP = gamblersProblemMDP.applyFinitePolicy(maxBetPolicy)
  val impliedValueFunction = impliedMRP.valueFunctionVector(gamma)
  val impliedValueFunctionStr = impliedMRP.valueFunctionToString(gamma)

  val optimalVfStr = valueFunctionToString(optimalValueFunction, Some(sortingFunction))

  val xs = nonTerminalStates.map(_.state.toDouble)
  val optimalValues = nonTerminalStates.map(optimalValueFunction.apply)
  val optimalActions = nonTerminalStates.map(nt => optimalPolicy.actionForState(nt.state).toDouble)
  val maxBetActions = nonTerminalStates.map(nt => maxBetPolicy.actionForState(nt.state).toDouble)

  val fig = Figure("Gambler's Problem Optimal Value Function and Max-Bet Policy Value Function")
  val p = fig.subplot(0)
  p += plot(xs, optimalValues, name = "Optimal Value Function")
  p += plot(xs, impliedValueFunction, name = "Max-Bet Policy Value Function")
  p.legend = true
  p.setYAxisDecimalTickUnits()

  val fig2 = Figure("Gambler's Problem Optimal Policy and Max Bet Policy")
  val p2 = fig2.subplot(0)
  p2 += plot(xs, optimalActions, style = '.', name = "Optimal Policy")
  p2 += plot(xs, maxBetActions, style = '+', name = "Max Bet Policy")
  p2.legend = true
  p2.setYAxisIntegerTickUnits()

  logger.info(f"Optimal Value Function:\n$optimalVfStr")
  logger.info(f"Optimal Policy:\n${optimalPolicy.printPolicy(_ < _)}")
  logger.info(f"Implied Value Function From Policy:\n$impliedValueFunctionStr")

  logger.info("Finishing App execution")

}
