# Import test helpers
Code.require_file("test_helper.exs", __DIR__)

defmodule MetaLearningEmergenceTest do
  use ExUnit.Case

  @moduledoc """
  Tests the meta-learning system's ability to evolve from a simple state to a complex
  adaptive intelligence, following the principles of the emergent foundation model.

  This test simulates the progression from 0→1 to A→Z, where the system gradually
  develops awareness of its own learning patterns and adapts accordingly.
  """

  alias KernelShtf.MetaLearning
  import EmergenceTestHelper
  import EmergenceTestAssertions

  test "emergent foundation model: system evolves from simple to complex learning" do
    # ===== STAGE 1: THE SEED (0) =====
    seed_classifier = EmergenceTestStages.create_seed()
    verify_seed(seed_classifier, EmergenceTestConfigs.basic_vocabulary())

    # ===== STAGE 2: THE FIRST SPROUT (1) =====
    sprout_classifier = EmergenceTestStages.create_sprout(seed_classifier)
    verify_learning(seed_classifier, sprout_classifier, "good", "positive")

    # ===== STAGE 3: THE FIRST BRANCH (A → B, 1 → 2) =====
    branch_classifier = EmergenceTestStages.create_branch(sprout_classifier)

    # ===== STAGE 4: THE NETWORK OF GROWTH =====
    network_classifier = EmergenceTestStages.create_network(branch_classifier)

    # ===== STAGE 5: THE CANOPY (Emergent Complexity) =====
    verify_canopy_tests(network_classifier, EmergenceTestData.canopy_test_cases())

    # ===== STAGE 6: META-LEARNING (Self-modification) =====
    meta_trained_classifier = EmergenceTestStages.create_meta_learner(network_classifier)
    verify_meta_learning(
      network_classifier,
      meta_trained_classifier,
      EmergenceTestData.negation_test_cases()
    )

    # ===== STAGE 7: THE ECOSYSTEM (Complex Interactions) =====
    ecosystem_classifier = EmergenceTestStages.create_ecosystem(meta_trained_classifier)
    verify_meta_learning(
      meta_trained_classifier,
      ecosystem_classifier,
      EmergenceTestData.context_test_cases()
    )

    # ===== STAGE 8: META-AWARENESS (System Introspection) =====
    verify_feature_identification(
      ecosystem_classifier,
      ["excellent", "good", "terrible", "bad"]
    )

    # ===== STAGE 9: RECURSION (Z→A, Full Circle) =====
    reborn_classifier = EmergenceTestStages.create_reborn_classifier(ecosystem_classifier)
    {reborn_trained, seed_trained} = EmergenceTestStages.train_final_classifiers(reborn_classifier)

    # Verify the reborn classifier performance
    verify_reborn_performance(
      reborn_trained,
      seed_trained,
      EmergenceTestData.final_test_cases()
    )
  end
end
