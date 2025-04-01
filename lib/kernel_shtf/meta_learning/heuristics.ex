defmodule KernelShtf.MetaLearning.Heuristics do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning.Types.Learning
  alias KernelShtf.MetaLearning.Evidence
  alias KernelShtf.MetaLearning.Parameters
  alias KernelShtf.MetaLearning.Transformations

  def apply_meta_rules(heuristics, meta_rules, dataset) do
    # Meta-rules examine patterns in the learning process and modify heuristics
    Enum.reduce(meta_rules, {heuristics, dataset}, fn meta_rule,
                                                      {current_heuristics, current_dataset} ->
      fold meta_rule, with: {current_heuristics, current_dataset} do
        case(metaRule(pattern, transformation, priority)) ->
          # Check if the pattern matches current learning state
          if pattern_matches?(pattern, current_heuristics, current_dataset) do
            # Apply transformation to heuristics
            {transformed_heuristics, updated_dataset} =
              Transformations.apply_transformation(
                transformation,
                current_heuristics,
                current_dataset
              )

            {transformed_heuristics, updated_dataset}
          else
            {current_heuristics, current_dataset}
          end
      end
    end)
  end

  def self_modify_meta_rules(meta_rules, dataset) do
    # Meta-meta-learning: rules that modify the meta-rules
    evidence = Evidence.extract_learning_evidence(dataset)

    # Use Enum.map instead of fold with custom pattern matching
    new_meta_rules =
      Enum.map(meta_rules, fn meta_rule ->
        # Extract fields directly
        pattern = Map.get(meta_rule, :pattern)
        transformation = Map.get(meta_rule, :transformation)
        priority = Map.get(meta_rule, :priority)

        # Based on evidence, meta-rules can modify themselves
        if should_adapt_rule?(pattern, evidence) do
          # Create a variant of the rule with modified transformation
          new_transformation = Transformations.evolve_transformation(transformation, evidence)
          new_priority = Transformations.calculate_new_priority(priority, pattern, evidence)

          Learning.metaRule(pattern, new_transformation, new_priority)
        else
          # Keep the original rule
          meta_rule
        end
      end)

    {new_meta_rules, dataset}
  end

  # Apply heuristics to update parameters based on data
  def apply_heuristics(parameters, heuristics, batch) do
    # Extract data points from batch for processing
    data_points = extract_data_points(batch)

    # Process each data point to update parameters
    Enum.reduce(data_points, {parameters, batch}, fn data_point,
                                                      {current_params, current_batch} ->
      # Apply each heuristic that matches the data point
      Enum.reduce(heuristics, {current_params, current_batch}, fn heuristic, {params, batch} ->
        if heuristic_applies_to_data?(heuristic, data_point) do
          # Apply heuristic to update parameters
          update_parameters_for_data_point(params, heuristic, data_point, batch)
        else
          {params, batch}
        end
      end)
    end)
  end

  # Check if a pattern matches the current learning state
  def pattern_matches?(pattern, heuristics, dataset) do
    cond do
      is_function(pattern, 2) ->
        # Pattern is a function that takes heuristics and dataset
        pattern.(heuristics, dataset)

      is_map(pattern) && Map.has_key?(pattern, :condition) ->
        # Pattern has a condition function
        pattern.condition.(heuristics, dataset)

      is_map(pattern) && Map.has_key?(pattern, :heuristic_type) ->
        # Pattern matches heuristic types
        heuristic_type_exists?(pattern.heuristic_type, heuristics)

      is_map(pattern) && Map.has_key?(pattern, :dataset_property) ->
        # Pattern checks for dataset property
        dataset_has_property?(pattern.dataset_property, dataset)

      true ->
        false
    end
  end

  # Check if a specific heuristic type exists
  def heuristic_type_exists?(type, heuristics) do
    Enum.any?(heuristics, fn
      %{variant: :heuristic, condition: condition} ->
        condition_matches_type?(condition, type)

      _ ->
        false
    end)
  end

  # Check if a condition matches a type
  def condition_matches_type?(condition, type) do
    cond do
      is_map(condition) && Map.get(condition, :type) == type -> true
      is_atom(condition) && condition == type -> true
      is_binary(condition) && condition == to_string(type) -> true
      true -> false
    end
  end

  # Check if dataset has a specific property
  def dataset_has_property?(property, dataset) do
    fold dataset do
      case(dataset(data_points, statistics)) ->
        Map.has_key?(statistics, property)
    end
  end

  # Check if a heuristic applies to a specific data point
  def heuristic_applies_to_data?(heuristic, data_point) do
    # Direct map access instead of fold pattern matching
    if is_map(heuristic) && Map.has_key?(heuristic, :variant) && heuristic.variant == :heuristic do
      condition = heuristic[:condition]

      if is_map(data_point) && Map.has_key?(data_point, :variant) && data_point.variant == :dataPoint do
        features = data_point[:features]
        label = data_point[:label]

        # Check if condition matches features/label
        cond do
          is_function(condition, 2) ->
            condition.(features, label)

          is_function(condition, 1) ->
            # Handle both atom and string feature categories
            category = Map.get(features, :category) || Map.get(features, "category")
            condition.(category)

          is_map(condition) && Map.has_key?(condition, :feature_matcher) ->
            condition.feature_matcher.(features)

          is_map(condition) && Map.has_key?(condition, :label_matcher) ->
            condition.label_matcher.(label)

          # Handle simple parameter name match
          is_map(condition) && Map.has_key?(condition, :param_name) ->
            true  # We'll check specific param matching in update_parameter

          is_atom(condition) || is_binary(condition) ->
            # Simple condition that just checks parameter presence
            true

          true ->
            # Default to true for simple conditions that don't explicitly match data
            true
        end
      else
        false
      end
    else
      false
    end
  end

  # Update parameters based on a data point and heuristic
  def update_parameters_for_data_point(parameters, heuristic, data_point, batch) do
    fold heuristic do
      case(heuristic(condition, action, confidence)) ->
        fold data_point do
          case(dataPoint(features, label)) ->
            # Apply the parameter update action with the data point context
            Enum.reduce(parameters, {[], batch}, fn param, {acc_params, acc_batch} ->
              {updated_param, new_batch} =
                update_parameter(
                  param,
                  action,
                  confidence,
                  features,
                  label,
                  acc_batch
                )

              {[updated_param | acc_params], new_batch}
            end)
        end
    end
  end

  # Update a single parameter based on a heuristic action
  def update_parameter(param, action, confidence, features, label, batch) do
    # Direct map access instead of fold pattern matching
    name = param[:name]
    value = param[:value]
    update_rate = param[:update_rate]

    # Only log parameter updates when significant changes occur
    debug_enabled = System.get_env("DEBUG_META_LEARNING") == "true"

    # Apply action to calculate parameter delta
    delta_result = calculate_parameter_delta(action, name, value, features, label, batch)

    # Ensure delta_result is properly processed
    {delta_value, updated_batch} = case delta_result do
      {delta, new_batch} when is_number(delta) ->
        {delta, new_batch}
      delta when is_number(delta) ->
        {delta, batch}
      _ ->
        {0.0, batch}  # Default case for safety
    end

    # Scale update by confidence and learning rate
    scaled_delta = delta_value * confidence * update_rate

    # Force a more significant update for learning progress
    effective_delta = if abs(scaled_delta) < 0.001, do: sign(scaled_delta) * 0.001, else: scaled_delta

    # Create updated parameter with the new value
    updated_param = %{param | value: value + effective_delta}

    # Only log significant updates to reduce verbosity
    if debug_enabled && abs(effective_delta) > 0.01 do
      IO.puts("Param update: #{name} = #{value} → #{value + effective_delta} (Δ: #{effective_delta})")
    end

    {updated_param, updated_batch}
  end

  # Helper function to get the sign of a number
  def sign(x) when x > 0, do: 1.0
  def sign(x) when x < 0, do: -1.0
  def sign(_), do: 0.0

  # Calculate the delta for a parameter update
  def calculate_parameter_delta(action, name, current_value, features, label, batch) do
    cond do
      is_function(action, 6) ->
        # Action takes name, value, rate, features, label, batch
        delta = action.(name, current_value, 0.1, features, label, batch)
        {delta, batch}

      is_function(action, 5) ->
        # Action takes name, value, features, label, batch
        delta = action.(name, current_value, features, label, batch)
        {delta, batch}

      is_function(action, 3) ->
        # Action takes name, value, features
        delta = action.(name, current_value, features)
        {delta, batch}

      is_function(action, 2) ->
        # Action takes name, value
        delta = action.(name, current_value)
        {delta, batch}

      is_map(action) && Map.has_key?(action, :gradient_function) ->
        # Action computes gradient based on features and label
        gradient = action.gradient_function.(features, label, current_value)
        {gradient, batch}

      is_map(action) && Map.has_key?(action, :error_function) ->
        # Action computes error and derives update from it
        predicted = apply_prediction(current_value, features)
        error = action.error_function.(predicted, label)
        # Negative error for gradient descent
        {-error, batch}

      is_map(action) && Map.has_key?(action, :delta) ->
        # Direct delta specification
        {action.delta, batch}

      # FIXED DEFAULT CASE - Make learning more adaptive and consider context
      true ->
        # Check if this word appears in features
        word_count = Map.get(features, name, 0)
        category = Map.get(features, :category) || Map.get(features, "category")

        # Make sure to use strings for label comparison
        label_str = if is_atom(label), do: Atom.to_string(label), else: label
        category_str = if is_atom(category), do: Atom.to_string(category), else: category

        # Get category-specific configuration if available
        config = extract_config_from_batch(batch)
        key_terms = get_key_terms_for_category(config, label_str)

        cond do
          # Word appears and matches correct category - strong positive signal
          word_count > 0 && category_str == label_str ->
            # Boost if it's a key term for this category
            boost = if name in key_terms, do: 1.5, else: 1.0
            {0.5 * boost, batch}

          # Word appears but in wrong category - strong negative signal
          word_count > 0 && category_str != label_str ->
            # Stronger negative for key terms in wrong categories
            penalty = if in_category_terms?(name, label_str, batch), do: 1.5, else: 1.0
            {-0.3 * penalty, batch}

          # Expected term missing in this category - modest decay
          word_count == 0 && in_category_terms?(name, label_str, batch) ->
            {-0.1, batch}

          # Unused term - slight decay
          true ->
            # More aggressive decay for advanced stages
            decay = case get_learning_stage(batch) do
              stage when stage > 3 -> -0.04  # More aggressive in later stages
              _ -> -0.02
            end
            {decay, batch}
        end
    end
  end

  # Helper to extract config from batch
  def extract_config_from_batch(batch) do
    cond do
      is_map(batch) && Map.has_key?(batch, :statistics) &&
        is_map(batch.statistics) && Map.has_key?(batch.statistics, :config) ->
        batch.statistics.config
      is_map(batch) && Map.has_key?(batch, :config) ->
        batch.config
      true ->
        %{key_terms: %{}}
    end
  end

  # Helper to get key terms for a category
  def get_key_terms_for_category(config, category) do
    case Map.get(config, :key_terms, %{}) do
      key_terms when is_map(key_terms) ->
        Map.get(key_terms, category, [])
      _ ->
        []
    end
  end

  # Helper to estimate the learning stage based on batch statistics
  def get_learning_stage(batch) do
    if is_map(batch) && Map.has_key?(batch, :statistics) do
      case Map.get(batch.statistics, :learning_stage) do
        nil ->
          # Estimate stage based on other indicators
          parameter_count = Map.get(batch.statistics, :parameter_count, 0)
          meta_rule_count = Map.get(batch.statistics, :meta_rule_count, 0)
          cond do
            meta_rule_count > 2 -> 6  # Meta-learning stage
            parameter_count > 20 -> 4  # Network stage
            parameter_count > 10 -> 3  # Branch stage
            true -> 2  # Sprout stage
          end
        stage when is_number(stage) -> stage
        _ -> 2  # Default to sprout stage
      end
    else
      2  # Default to sprout stage
    end
  end

  # Helper function to check if a word is in a category's key terms
  def in_category_terms?(word, category, batch) do
    # Try to extract config from batch or use default
    config =
      cond do
        is_map(batch) && Map.has_key?(batch, :statistics) &&
        Map.has_key?(batch.statistics, :config) ->
          batch.statistics.config
        true ->
          %{key_terms: %{}}
      end

    # Check if word is in the category's key terms
    case Map.get(config, :key_terms, %{}) do
      key_terms when is_map(key_terms) ->
        terms = Map.get(key_terms, category, [])
        word in terms
      _ ->
        false
    end
  end

  # Apply a parameter to make a prediction
  def apply_prediction(parameter_value, features) do
    # Simple linear combination for demonstration
    # In practice, this would depend on the model architecture
    if is_map(features) && is_number(parameter_value) do
      # Dot product for vector features
      Enum.sum(for {_k, v} <- features, do: v * parameter_value)
    else
      # Default simple multiplication
      parameter_value * 1.0
    end
  end

  # Extract data points from a batch
  def extract_data_points(batch) do
    cond do
      # Handle proper batch structure
      is_map(batch) && Map.has_key?(batch, :variant) && batch.variant == :batch &&
      Map.has_key?(batch, :data_subset) ->
        batch.data_subset

      # Handle list of data points directly
      is_list(batch) ->
        batch

      # Handle dataset structure
      is_map(batch) && Map.has_key?(batch, :variant) && batch.variant == :dataset &&
      Map.has_key?(batch, :data_points) ->
        batch.data_points

      # Fallback for any other structure
      true ->
        []
    end
  end

  # Check if a meta-rule should adapt based on learning evidence
  def should_adapt_rule?(pattern, evidence) do
    fold evidence do
      case(learningEvidence(patterns, effectiveness, surprise)) ->
        # Calculate adaptation score based on evidence
        adaptation_score = calculate_adaptation_score(pattern, patterns, effectiveness, surprise)

        # Adapt rule if score exceeds threshold
        adaptation_score > 0.7
    end
  end

  # Calculate an adaptation score for a meta-rule
  def calculate_adaptation_score(pattern, patterns, effectiveness, surprise) do
    # Base factors for adaptation decision
    base_factors = [
      # Adapt more when learning is ineffective
      {0.4, 1.0 - effectiveness},

      # Adapt more when surprises are detected
      {0.3, surprise.score},

      # Adapt more when the pattern matches current conditions
      {0.3, pattern_match_score(pattern, patterns)}
    ]

    # Calculate weighted score
    Enum.map(base_factors, fn {weight, score} -> weight * score end)
    |> Enum.sum()
  end

  # Calculate how well a pattern matches current conditions
  def pattern_match_score(pattern, detected_patterns) do
    cond do
      is_map(pattern) && Map.has_key?(pattern, :heuristic_type) ->
        # Check for patterns related to specified heuristic type
        case Map.get(detected_patterns, pattern.heuristic_type) do
          nil ->
            0.0

          %{type: pattern_type} ->
            if Map.get(pattern, :expected_pattern) == pattern_type, do: 1.0, else: 0.2

          _ ->
            0.0
        end

      is_map(pattern) && Map.has_key?(pattern, :param_name) ->
        # Check for patterns in specific parameter
        case Map.get(detected_patterns, pattern.param_name) do
          nil -> 0.0
          %{} = param_pattern -> pattern_type_match_score(pattern, param_pattern)
          _ -> 0.0
        end

      # Default modest score for generic patterns
      true ->
        0.2
    end
  end

  # Calculate match score between expected and detected pattern types
  def pattern_type_match_score(expected_pattern, detected_pattern) do
    expected_type = Map.get(expected_pattern, :expected_pattern)
    actual_type = Map.get(detected_pattern, :type)

    cond do
      expected_type == actual_type -> 1.0
      expected_type == :any -> 0.5
      true -> 0.1
    end
  end
end
