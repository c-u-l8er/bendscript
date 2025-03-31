defmodule KernelShtf.MetaLearning.Transformations do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning.Types.Learning
  alias KernelShtf.MetaLearning.Parameters

  # Apply a transformation to modify heuristics
  def apply_transformation(transformation, heuristics, dataset) do
    fold transformation, with: {heuristics, dataset} do
      case(transformation) ->
        # Execute the transformation function
        result = execute_transformation(transformation, heuristics, dataset)

        {Enum.at(result, 0), Enum.at(result, 1)}
    end
  end

  # Execute a transformation function on heuristics
  def execute_transformation(transformation, heuristics, dataset) do
    cond do
      is_function(transformation, 2) ->
        # Transformation is a function that takes heuristics and dataset
        new_heuristics = transformation.(heuristics, dataset)
        {new_heuristics, dataset}

      is_map(transformation) && Map.has_key?(transformation, :function) ->
        # Transformation has a function property
        transformation.function.(heuristics, dataset)

      is_map(transformation) && Map.has_key?(transformation, :add_heuristic) ->
        # Transformation adds a new heuristic
        {[transformation.add_heuristic | heuristics], dataset}

      is_map(transformation) && Map.has_key?(transformation, :modify_confidence) ->
        # Transformation modifies confidence of existing heuristics
        new_heuristics =
          modify_heuristic_confidence(
            heuristics,
            transformation.modify_confidence
          )

        {new_heuristics, dataset}

      true ->
        # Default: no change
        {heuristics, dataset}
    end
  end

  # Modify confidence levels of heuristics
  def modify_heuristic_confidence(heuristics, confidence_delta) do
    Enum.map(heuristics, fn
      %{variant: :heuristic, condition: condition, action: action, confidence: confidence} ->
        # Adjust confidence level, ensuring it stays between 0 and 1
        new_confidence = max(0.0, min(1.0, confidence + confidence_delta))
        Learning.heuristic(condition, action, new_confidence)

      other ->
        other
    end)
  end

  # Evolve a transformation based on learning evidence
  def evolve_transformation(transformation, evidence) do
    fold evidence do
      case(learningEvidence(patterns, effectiveness, surprise)) ->
        # Determine how to evolve the transformation
        if effectiveness < 0.3 && surprise.score > 0.7 do
          # Major transformation change needed
          create_novel_transformation(transformation, patterns, surprise)
        else
          # Moderate improvements to existing transformation
          refine_transformation(transformation, patterns, effectiveness)
        end
    end
  end

  # Create a completely new transformation
  def create_novel_transformation(original, patterns, surprise) do
    # Identify the most surprising pattern
    {most_surprising_param, pattern_info} =
      Enum.max_by(
        patterns,
        fn {_param, info} ->
          case info do
            %{type: :oscillation} -> 0.9
            %{type: :divergence} -> 1.0
            %{type: :convergence} -> 0.3
            _ -> 0.5
          end
        end,
        fn -> {nil, %{}} end
      )

    case pattern_info do
      %{type: :oscillation} ->
        # Create momentum-based transformation to dampen oscillations
        create_momentum_transformation(most_surprising_param)

      %{type: :divergence} ->
        # Create stabilizing transformation
        create_stabilizing_transformation(most_surprising_param, pattern_info)

      _ ->
        # Default to refined version of original
        refine_transformation(original, patterns, 0.5)
    end
  end

  # Create a momentum-based transformation to handle oscillations
  def create_momentum_transformation(param_name) do
    # Create a function that applies momentum to parameter updates
    momentum_function = fn heuristics, dataset ->
      # Find heuristics affecting this parameter
      target_heuristics =
        Enum.filter(heuristics, fn
          %{variant: :heuristic, condition: condition} ->
            Parameters.parameter_matches_condition?(param_name, condition)

          _ ->
            false
        end)

      # Modify those heuristics to incorporate momentum
      updated_heuristics =
        Enum.map(heuristics, fn heuristic ->
          if Enum.member?(target_heuristics, heuristic) do
            add_momentum_to_heuristic(heuristic)
          else
            heuristic
          end
        end)

      {updated_heuristics, dataset}
    end

    # Return the transformation
    Learning.metaRule(
      %{param_name: param_name, expected_pattern: :oscillation},
      momentum_function,
      0.9
    )
  end

  # Add momentum to a heuristic
  def add_momentum_to_heuristic(heuristic) do
    fold heuristic do
      case(heuristic(condition, action, confidence)) ->
        # Create a new action that incorporates momentum
        new_action = fn name, value, update_rate, dataset ->
          # Get previous updates
          previous_updates = Parameters.get_previous_updates(name, dataset)

          # Calculate raw update using original action
          raw_update = Parameters.execute_action(action, name, value, update_rate, dataset)
          current_update = elem(raw_update, 0) - value

          # Apply momentum
          momentum = 0.8

          averaged_update =
            if previous_updates == [] do
              current_update
            else
              # Weighted average of current and previous updates
              previous_avg = Enum.sum(previous_updates) / length(previous_updates)
              current_update * (1 - momentum) + previous_avg * momentum
            end

          # Return new value
          {value + averaged_update, elem(raw_update, 1)}
        end

        # Return modified heuristic
        Learning.heuristic(condition, new_action, confidence)
    end
  end

  # Create a stabilizing transformation for diverging parameters
  def create_stabilizing_transformation(param_name, pattern_info) do
    # Extract divergence rate
    divergence_rate = Map.get(pattern_info, :rate, 1.0)

    # Create a function that adds stability constraints
    stabilize_function = fn heuristics, dataset ->
      # Find heuristics affecting this parameter
      target_heuristics =
        Enum.filter(heuristics, fn
          %{variant: :heuristic, condition: condition} ->
            Parameters.parameter_matches_condition?(param_name, condition)

          _ ->
            false
        end)

      # Modify those heuristics to add stability
      updated_heuristics =
        Enum.map(heuristics, fn heuristic ->
          if Enum.member?(target_heuristics, heuristic) do
            add_stability_to_heuristic(heuristic, divergence_rate)
          else
            heuristic
          end
        end)

      {updated_heuristics, dataset}
    end

    # Return the transformation
    Learning.metaRule(
      %{param_name: param_name, expected_pattern: :divergence},
      stabilize_function,
      # High priority for divergence
      1.0
    )
  end

  # Add stability constraints to a heuristic (continued)
  def add_stability_to_heuristic(heuristic, divergence_rate) do
    fold heuristic do
      case(heuristic(condition, action, confidence)) ->
        # Create a new action with stability constraints
        new_action = fn name, value, update_rate, dataset ->
          # Calculate raw update
          raw_update = Parameters.execute_action(action, name, value, update_rate, dataset)
          calculated_value = elem(raw_update, 0)

          # Calculate change
          change = calculated_value - value

          # Apply stability constraint - limit maximum change based on divergence
          max_change = value * 0.1 / divergence_rate
          clamped_change = max(-max_change, min(max_change, change))

          # Return stabilized value
          {value + clamped_change, elem(raw_update, 1)}
        end

        # Return modified heuristic with reduced confidence
        # Reduce confidence in diverging heuristic
        new_confidence = min(0.5, confidence)
        Learning.heuristic(condition, new_action, new_confidence)
    end
  end

  # Refine an existing transformation
  def refine_transformation(transformation, patterns, effectiveness) do
    fold transformation do
      case(transformation) ->
        # Apply small improvements to the transformation
        if is_function(transformation) do
          # Wrap function with monitoring logic
          create_monitored_transformation(transformation, effectiveness)
        else
          # Try to extract and enhance the core functionality
          fold transformation do
            case(metaRule(pattern, inner_transformation, priority)) ->
              # Adjust priority based on effectiveness
              new_priority = adjust_priority(priority, effectiveness)

              # Enhance the pattern to be more specific
              new_pattern = enhance_pattern(pattern, patterns)

              # Return refined meta rule
              Learning.metaRule(new_pattern, inner_transformation, new_priority)
          end
        end
    end
  end

  # Create a monitored version of a transformation function
  def create_monitored_transformation(transform_fn, effectiveness) do
    fn heuristics, dataset ->
      # Apply the original transformation
      {transformed_heuristics, updated_dataset} = transform_fn.(heuristics, dataset)

      # Add monitoring data to dataset
      monitoring_info = %{
        transformation_applied: true,
        timestamp: DateTime.utc_now(),
        pre_transform_count: length(heuristics),
        post_transform_count: length(transformed_heuristics),
        effectiveness: effectiveness
      }

      # Update dataset with monitoring info
      dataset_with_monitoring =
        fold updated_dataset do
          case(dataset(data_points, statistics)) ->
            # Add monitoring info to statistics
            monitoring_history = Map.get(statistics, :transformation_history, [])

            new_statistics =
              Map.put(
                statistics,
                :transformation_history,
                [monitoring_info | monitoring_history]
              )

            Learning.dataset(data_points, new_statistics)
        end

      {transformed_heuristics, dataset_with_monitoring}
    end
  end

  # Enhance a pattern to be more specific based on observed patterns
  def enhance_pattern(pattern, observed_patterns) do
    if is_map(pattern) do
      # Add more specific matching conditions based on observed patterns
      matching_keys =
        for {param_name, pattern_info} <- observed_patterns,
            Map.get(pattern, :param_name) == param_name do
          pattern_info
        end

      if matching_keys != [] do
        # Add the most common pattern type to expected patterns
        pattern_info = hd(matching_keys)
        Map.put(pattern, :expected_pattern, Map.get(pattern_info, :type, :any))
      else
        pattern
      end
    else
      # Convert non-map patterns to maps for extensibility
      %{original_pattern: pattern}
    end
  end

  # Adjust priority based on effectiveness
  def adjust_priority(priority, effectiveness) do
    # Increase priority if rule is working well
    if effectiveness > 0.7 do
      min(1.0, priority * 1.2)
    else
      # Decrease priority if rule is not effective
      max(0.1, priority * 0.8)
    end
  end

  # Calculate new priority for a meta-rule
  def calculate_new_priority(priority, pattern, evidence) do
    fold evidence do
      case(learningEvidence(patterns, effectiveness, surprise)) ->
        # Calculate match score
        match_score = pattern_match_score(pattern, patterns)

        # Adjust priority based on effectiveness and match
        cond do
          match_score > 0.8 && effectiveness > 0.7 ->
            # Pattern matches and is effective - increase priority
            min(1.0, priority * 1.2)

          match_score > 0.8 && effectiveness < 0.3 ->
            # Pattern matches but is ineffective - decrease priority
            max(0.1, priority * 0.5)

          match_score < 0.2 ->
            # Pattern doesn't match current conditions - lower priority
            max(0.1, priority * 0.7)

          true ->
            # Default - slight adjustment based on effectiveness
            max(0.1, min(1.0, priority * (0.8 + effectiveness * 0.4)))
        end
    end
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
