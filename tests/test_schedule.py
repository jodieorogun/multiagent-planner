import unittest

from agents.critic import CriticAgent
from agents.fitness import FitnessAgent
from core.models import PlanningBrief, SportsCommitment, WorkoutGoal


class ScheduleMetadataTests(unittest.TestCase):
    def test_schedule_keeps_event_metadata_and_surfaces_conflicts(self):
        brief = PlanningBrief(
            sports_commitments=[
                SportsCommitment("Match", "Wednesday", 5.0),
                SportsCommitment("Training", "Wednesday", 5.0),
            ],
            workout_goals=[WorkoutGoal("Gym", 1, 1.0)],
        )

        schedule = FitnessAgent().run(brief)

        self.assertEqual(schedule.days["Wednesday"][0].priority, "fixed")
        self.assertEqual(schedule.days["Wednesday"][0].source, "user")
        self.assertTrue(schedule.conflicts)

    def test_critic_reports_the_peak_day(self):
        brief = PlanningBrief(
            sports_commitments=[SportsCommitment("Match", "Wednesday", 3.0)],
            workout_goals=[WorkoutGoal("Gym", 2, 2.0)],
        )
        schedule = FitnessAgent().run(brief)

        assessment = CriticAgent(lambda features: 1).run(brief, schedule)

        self.assertEqual(assessment.peak_day, "Wednesday")
        self.assertGreater(assessment.daily_load["Wednesday"], 0)


if __name__ == "__main__":
    unittest.main()
