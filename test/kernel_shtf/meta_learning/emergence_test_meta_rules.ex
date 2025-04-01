defmodule EmergenceTestMetaRules do
  @moduledoc """
  Meta-rules for the emergence test
  """

  alias KernelShtf.MetaLearning
  alias KernelShtf.MetaLearning.Types.Learning

  # Create a negation-handling meta-rule
  def negation_meta_rule do
    MetaLearning.new_meta_rule(
      %{heuristic_type: :feature_negation},
      fn heuristics, dataset ->
        negation_heuristic = MetaLearning.new_heuristic(
          fn param_name -> Enum.member?(["good", "bad", "like", "dislike", "love", "hate"], param_name) end,
          fn name, value, _rate, features, label, _batch ->
            # If "not" appears before a word, reverse its effect
            text = Map.get(features, :original_text, "")
            if text && String.contains?(text, "not #{name}") do
              -value * 2.0  # Reverse and amplify
            else
              0.0
            end
          end,
          0.85
        )

        {[negation_heuristic | heuristics], dataset}
      end,
      0.9  # High priority
    )
  end

  # Create a context-aware meta-rule
  def context_meta_rule do
    MetaLearning.new_meta_rule(
      %{heuristic_type: :context_awareness},
      fn heuristics, dataset ->
        # This meta-rule creates a context-aware heuristic that looks at word combinations
        context_heuristic = MetaLearning.new_heuristic(
          fn _ -> true end,  # Applies to all parameters
          fn name, value, _rate, features, label, _batch ->
            text = Map.get(features, :original_text, "")
            if text do
              cond do
                # Product quality context
                (name in ["good", "bad", "excellent", "terrible"]) &&
                (String.contains?(text, "quality") || String.contains?(text, "product")) ->
                  value * 1.5  # Amplify in product context

                # Price context
                (name in ["expensive", "cheap"]) &&
                (String.contains?(text, "price") || String.contains?(text, "cost")) ->
                  if label in ["very_positive", "positive"] && name == "cheap" ||
                     label in ["negative", "very_negative"] && name == "expensive" do
                    value * 2.0  # Amplify when aligned with sentiment
                  else
                    value * 0.5  # Reduce when contradicting sentiment
                  end

                # Service context
                (name in ["fast", "slow"]) &&
                (String.contains?(text, "service") || String.contains?(text, "delivery")) ->
                  value * 1.8  # Amplify in service context

                true -> 0.0  # No adjustment
              end
            else
              0.0
            end
          end,
          0.9
        )

        {[context_heuristic | heuristics], dataset}
      end,
      0.95  # Very high priority
    )
  end

  # Create a self-optimizing adaptive meta-rule
  def adaptive_meta_rule do
    MetaLearning.new_meta_rule(
      %{heuristic_type: :adaptive_weights},
      fn heuristics, dataset ->
        # This meta-rule analyzes the dataset and adjusts learning rates dynamically
        # Manually extract data_points and statistics instead of using fold
        data_points = Map.get(dataset, :data_points, [])
        statistics = Map.get(dataset, :statistics, %{})

        # Check if we have stats on parameter effectiveness
        if Map.has_key?(statistics, :param_effectiveness) do
          effectiveness = Map.get(statistics, :param_effectiveness, %{})

          # Create a new heuristic that adjusts update rates based on effectiveness
          adaptive_heuristic = MetaLearning.new_heuristic(
            fn param_name -> true end, # Apply to all parameters
            fn name, value, rate, features, label, _batch ->
              # Get effectiveness score for this parameter
              score = Map.get(effectiveness, name, 0.5)

              # Adjust learning rate based on effectiveness
              if score > 0.7 do
                # Very effective parameter - increase rate
                rate * 1.2
              else
                if score < 0.3 do
                  # Ineffective parameter - decrease rate
                  rate * 0.8
                else
                  # Maintain rate
                  rate
                end
              end
            end,
            0.9
          )
          {[adaptive_heuristic | heuristics], dataset}
        else
          {heuristics, dataset}
        end
      end,
      0.98  # Highest priority
    )
  end
end
