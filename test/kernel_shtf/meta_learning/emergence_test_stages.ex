defmodule EmergenceTestStages do
  @moduledoc """
  Individual test stages for the emergence test
  """

  alias KernelShtf.MetaLearning
  import EmergenceTestHelper

  # STAGE 1: THE SEED - Start with minimal structure and potential
  def create_seed do
    vocabulary = EmergenceTestConfigs.basic_vocabulary()
    categories = EmergenceTestConfigs.basic_categories()

    TextClassifier.new(vocabulary, categories)
  end

  # STAGE 2: THE FIRST SPROUT - Initial learning on simple patterns
  def create_sprout(seed_classifier) do
    IO.puts("DEBUG: Starting training for sprout stage")
    IO.puts("DEBUG: Initial weights: #{inspect(extract_parameter_values(seed_classifier.model.parameters))}")

    # Use the simple_docs data
    docs = EmergenceTestData.simple_docs()
    IO.puts("DEBUG: Training documents: #{inspect(docs)}")

    # Train with debug
    trained = TextClassifier.train(seed_classifier, docs, 5)

    IO.puts("DEBUG: Final weights: #{inspect(extract_parameter_values(trained.model.parameters))}")
    trained
  end

  # STAGE 3: THE FIRST BRANCH - Introducing complexity
  def create_branch(sprout_classifier) do
    TextClassifier.train(sprout_classifier, EmergenceTestData.complex_docs(), 5)
  end

  # STAGE 4: THE NETWORK OF GROWTH - Expanded vocabulary and structure
  def create_network(branch_classifier) do
    vocabulary = EmergenceTestConfigs.expanded_vocabulary()
    categories = EmergenceTestConfigs.expanded_categories()
    config = EmergenceTestConfigs.expanded_config()

    network_classifier = copy_learned_weights(
      TextClassifier.new(vocabulary, categories, config),
      branch_classifier
    )

    TextClassifier.train(network_classifier, EmergenceTestData.expanded_docs(), 10)
  end

  # STAGE 6: META-LEARNING - Self-modification with negation
  def create_meta_learner(network_classifier) do
    # Add the meta-rule to the model
    meta_classifier = %{
      network_classifier |
      model: %{
        network_classifier.model |
        meta_rules: [EmergenceTestMetaRules.negation_meta_rule() | network_classifier.model.meta_rules]
      }
    }

    TextClassifier.train(meta_classifier, EmergenceTestData.negation_docs(), 5)
  end

  # STAGE 7: THE ECOSYSTEM - Complex interactions with context
  def create_ecosystem(meta_trained_classifier) do
    # Add the context meta-rule
    ecosystem_classifier = %{
      meta_trained_classifier |
      model: %{
        meta_trained_classifier.model |
        meta_rules: [EmergenceTestMetaRules.context_meta_rule() | meta_trained_classifier.model.meta_rules]
      }
    }

    TextClassifier.train(ecosystem_classifier, EmergenceTestData.context_docs(), 5)
  end

  # STAGE 9: RECURSION - Create a new optimized classifier
  def create_reborn_classifier(ecosystem_classifier) do
    # Extract learned weights
    optimal_weights = extract_learned_weights(ecosystem_classifier)

    # Get learned initialization function
    learned_initialization = EmergenceTestConfigs.create_learned_initialization(optimal_weights)

    # Get vocabulary and categories
    reborn_vocabulary = EmergenceTestConfigs.reborn_vocabulary()
    expanded_categories = EmergenceTestConfigs.expanded_categories()

    # Get optimal config
    optimal_config = EmergenceTestConfigs.create_optimal_config(optimal_weights)

    # Build parameters list using the custom initialization strategy
    parameters = Enum.map(reborn_vocabulary, fn word -> learned_initialization.(word) end)

    # Create base heuristics
    base_heuristics = ecosystem_classifier.model.heuristics

    # Create model with learned meta-rules
    optimal_model = MetaLearning.new_model(
      parameters,
      base_heuristics,
      [EmergenceTestMetaRules.adaptive_meta_rule() | ecosystem_classifier.model.meta_rules]
    )

    # Create the newly evolved classifier
    TextClassifier.TextData.classifier(
      optimal_model,
      reborn_vocabulary,
      expanded_categories,
      optimal_config
    )
  end

  # Train the reborn classifier and seed for comparison
  def train_final_classifiers(reborn_classifier) do
    # Train reborn classifier
    reborn_trained = TextClassifier.train(reborn_classifier, EmergenceTestData.minimal_training(), 3)

    # Create and train seed classifier
    seed_with_vocab = TextClassifier.new(
      EmergenceTestConfigs.reborn_vocabulary(),
      EmergenceTestConfigs.expanded_categories()
    )
    seed_trained = TextClassifier.train(seed_with_vocab, EmergenceTestData.minimal_training(), 3)

    {reborn_trained, seed_trained}
  end
end
