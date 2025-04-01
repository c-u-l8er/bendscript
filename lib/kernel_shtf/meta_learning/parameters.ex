defmodule KernelShtf.MetaLearning.Parameters do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning.Types.Learning

  # Adjust parameters based on heuristics
  def adjust_parameters(parameters, heuristics, dataset) do
    # Group parameters by their characteristics for batch processing
    {fast_params, slow_params} = group_parameters_by_update_rate(parameters)

    # Apply different update strategies based on parameter types
    {updated_fast_params, dataset} = apply_fast_updates(fast_params, heuristics, dataset)
    {updated_slow_params, dataset} = apply_slow_updates(slow_params, heuristics, dataset)

    # Combine updated parameters
    {updated_fast_params ++ updated_slow_params, dataset}
  end

  # Group parameters by their update rate
  def group_parameters_by_update_rate(parameters) do
    Enum.split_with(parameters, fn
      %{variant: :parameter, update_rate: rate} -> rate > 0.5
      _ -> false
    end)
  end

  # Apply faster updates to parameters that change frequently
  def apply_fast_updates(parameters, heuristics, dataset) do
    Enum.reduce(parameters, {[], dataset}, fn param, {acc_params, acc_dataset} ->
      fold param, with: acc_dataset do
        case(parameter(name, value, update_rate)) ->
          # Find matching heuristic for this parameter
          matching_heuristic = find_matching_heuristic(name, heuristics)

          # Apply heuristic to update parameter value
          {new_value, updated_dataset} =
            apply_heuristic_to_parameter(
              matching_heuristic,
              name,
              value,
              update_rate,
              acc_dataset
            )

          # Create updated parameter
          new_param = Learning.parameter(name, new_value, update_rate)

          {[new_param | acc_params], updated_dataset}
      end
    end)
  end

  # Apply slower, more conservative updates to stable parameters
  def apply_slow_updates(parameters, heuristics, dataset) do
    Enum.reduce(parameters, {[], dataset}, fn param, {acc_params, acc_dataset} ->
      fold param, with: acc_dataset do
        case(parameter(name, value, update_rate)) ->
          # Use more conservative update strategy with momentum
          matching_heuristic = find_matching_heuristic(name, heuristics)

          # Get previous updates from dataset statistics
          previous_updates = get_previous_updates(name, acc_dataset)

          # Apply heuristic with momentum
          {new_value, updated_dataset} =
            apply_heuristic_with_momentum(
              matching_heuristic,
              name,
              value,
              update_rate,
              previous_updates,
              acc_dataset
            )

          new_param = Learning.parameter(name, new_value, update_rate)

          {[new_param | acc_params], updated_dataset}
      end
    end)
  end

  # Find a heuristic that matches the parameter name
  def find_matching_heuristic(param_name, heuristics) do
    Enum.find(heuristics, fn
      %{variant: :heuristic, condition: condition} ->
        parameter_matches_condition?(param_name, condition)

      _ ->
        false
    end)
  end

  # Check if a parameter name matches a heuristic condition
  def parameter_matches_condition?(param_name, condition) do
    cond do
      is_function(condition) -> condition.(param_name)
      is_map(condition) -> Map.get(condition, :param_name) == param_name
      is_binary(condition) -> condition == to_string(param_name)
      is_atom(condition) -> condition == param_name
      true -> false
    end
  end

  # Apply a heuristic to update a parameter value
  def apply_heuristic_to_parameter(heuristic, name, value, update_rate, dataset) do
    case heuristic do
      %{variant: :heuristic, action: action, confidence: confidence} ->
        # Check what kind of action we have
        cond do
          # If it's a function, execute it directly
          is_function(action) ->
            result = execute_action(action, name, value, update_rate, dataset)

            # Scale update by confidence level and learning rate
            {raw_delta, current_dataset} = result

            # Ensure we have a numeric value for the delta
            delta = cond do
              is_number(raw_delta) -> raw_delta
              is_tuple(raw_delta) && tuple_size(raw_delta) == 2 && is_number(elem(raw_delta, 0)) ->
                elem(raw_delta, 0)
              true -> 0.0
            end

            # Apply significant update to ensure learning happens
            new_value = value + delta * confidence * update_rate

            # Ensure we track parameter updates
            updated_dataset = store_parameter_update(name, delta, current_dataset)

            {new_value, updated_dataset}

          # If action is a map, handle it accordingly
          is_map(action) ->
            result = execute_action(action, name, value, update_rate, dataset)

            # Scale update by confidence level and learning rate
            {raw_delta, current_dataset} = result

            # Ensure we have a numeric value for the delta
            delta = cond do
              is_number(raw_delta) -> raw_delta
              is_tuple(raw_delta) && tuple_size(raw_delta) == 2 && is_number(elem(raw_delta, 0)) ->
                elem(raw_delta, 0)
              true -> 0.0
            end

            # Apply update
            new_value = value + delta * confidence * update_rate

            # Ensure we track parameter updates
            updated_dataset = store_parameter_update(name, delta, current_dataset)

            {new_value, updated_dataset}

          # For any other action type
          true ->
            # Use the action as is or provide a default
            {value, dataset}
        end

      # If no matching heuristic, return original value
      nil ->
        {value, dataset}

      # Any other case
      _ ->
        {value, dataset}
    end
  end

  # Apply heuristic with momentum based on previous updates
  def apply_heuristic_with_momentum(
        heuristic,
        name,
        value,
        update_rate,
        previous_updates,
        dataset
      ) do
    # Default momentum coefficient
    momentum = 0.9

    # Apply regular heuristic first
    {raw_update, dataset} =
      apply_heuristic_to_parameter(heuristic, name, value, update_rate, dataset)

    # Calculate change
    change = raw_update - value

    # Apply momentum based on previous updates
    averaged_change =
      if previous_updates == [] do
        change
      else
        # Weighted average of current change and previous changes
        previous_change = Enum.sum(previous_updates) / length(previous_updates)
        change * (1 - momentum) + previous_change * momentum
      end

    # Calculate new value
    new_value = value + averaged_change

    # Store the update in the dataset for future reference
    updated_dataset = store_parameter_update(name, change, dataset)

    {new_value, updated_dataset}
  end

  # Execute an action function to compute new parameter value
  def execute_action(action, name, value, update_rate, dataset) do
    cond do
      is_function(action, 4) ->
        new_value = action.(name, value, update_rate, dataset)
        {new_value, dataset}

      is_function(action, 2) ->
        new_value = action.(value, dataset)
        {new_value, dataset}

      is_map(action) && Map.has_key?(action, :function) ->
        action.function.(name, value, update_rate, dataset)

      is_map(action) && Map.has_key?(action, :delta) ->
        {value + action.delta, dataset}

      true ->
        # Default: no change
        {value, dataset}
    end
  end

  # Store parameter update in dataset for tracking momentum
  def store_parameter_update(name, change, dataset) do
    fold dataset do
      case(dataset(data_points, statistics)) ->
        # Get existing updates or initialize new list
        current_updates = Map.get(statistics, {:param_updates, name}, [])

        # Add new update, keep only recent updates (last 10)
        new_updates = Enum.take([change | current_updates], 10)

        # Update dataset statistics
        new_statistics = Map.put(statistics, {:param_updates, name}, new_updates)

        Learning.dataset(data_points, new_statistics)
    end
  end

  # Get previous parameter updates from dataset
  def get_previous_updates(name, dataset) do
    fold dataset do
      case(dataset(data_points, statistics)) ->
        Map.get(statistics, {:param_updates, name}, [])
    end
  end
end
