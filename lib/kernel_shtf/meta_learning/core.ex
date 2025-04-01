defmodule KernelShtf.MetaLearning.Core do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning.Types.Learning
  alias KernelShtf.MetaLearning.Parameters
  alias KernelShtf.MetaLearning.Heuristics
  alias KernelShtf.MetaLearning.Evidence
  alias KernelShtf.MetaLearning.Batching

  def train(initial_model, dataset, epochs) do
    if epochs <= 0 do
      initial_model
    else
      # First, let the meta-rules modify the model's learning approach
      {evolved_model, dataset} = evolve_learning_approach(initial_model, dataset)

      # Then apply the evolved model to learn from the dataset
      {trained_model, updated_dataset} = learn_epoch(evolved_model, dataset)

      # Recursive call to continue training
      train(trained_model, updated_dataset, epochs - 1)
    end
  end

  # This is where meta-pattern weaving happens - the system modifies its own learning rules
  def evolve_learning_approach(model, dataset) do
    fold model, with: dataset do
      case(model(parameters, heuristics, meta_rules)) ->
        # Apply meta-rules to transform heuristics
        {new_heuristics, dataset} = Heuristics.apply_meta_rules(heuristics, meta_rules, dataset)

        # Meta-rules can also modify themselves
        {evolved_meta_rules, dataset} = Heuristics.self_modify_meta_rules(meta_rules, dataset)

        # Adjust parameters based on new heuristics
        {new_parameters, dataset} =
          Parameters.adjust_parameters(parameters, new_heuristics, dataset)

        {Learning.model(new_parameters, new_heuristics, evolved_meta_rules), dataset}
    end
  end

  def learn_epoch(model, dataset) do
    debug_enabled = System.get_env("DEBUG_META_LEARNING") == "true"

    # Only log start of epoch when debugging is enabled
    if debug_enabled, do: IO.puts("Starting learn_epoch")

    # Extract batches from dataset
    batches = Batching.create_batches(dataset)

    # Log batch count only when debugging
    if debug_enabled, do: IO.puts("Created #{length(batches)} batches")

    # Learn from each batch sequentially
    {final_model, updated_batches} =
      Enum.reduce(batches, {model, []}, fn batch, {current_model, processed_batches} ->
        # Direct map access for model structure
        parameters = current_model.parameters
        heuristics = current_model.heuristics
        meta_rules = current_model.meta_rules

        # Add learning stage to batch statistics for adaptive behavior
        batch_with_stage = add_learning_stage_to_batch(batch, current_model, dataset)

        # Minimal logging of batch processing
        if debug_enabled do
          batch_size = length(batch[:data_subset] || [])
          IO.puts("Processing batch: #{batch_size} data points")
        end

        # Apply heuristics to determine how to update parameters
        {updated_parameters, updated_batch} =
          Heuristics.apply_heuristics(parameters, heuristics, batch_with_stage)

        # Check if parameters were actually updated - use deep comparison for values
        param_changed = parameters_changed?(parameters, updated_parameters)

        # If no parameters changed, apply more aggressive learning for key terms
        {final_parameters, final_batch} =
          if !param_changed do
            # Log only when applying boosted learning
            if debug_enabled, do: IO.puts("Applying boosted learning")
            # Apply boosted learning to ensure progress
            apply_boosted_learning(updated_parameters, heuristics, batch_with_stage)
          else
            {updated_parameters, updated_batch}
          end

        # Track learning statistics to update dataset
        final_batch = Batching.update_statistics(final_batch, parameters, final_parameters)

        # Save the updated batch
        {%{current_model | parameters: final_parameters},
         [final_batch | processed_batches]}
      end)

    # Merge batch statistics back into the dataset
    updated_dataset = merge_batch_statistics(dataset, updated_batches)

    {final_model, updated_dataset}
  end

  # Helper function to add learning stage to batch statistics
  def add_learning_stage_to_batch(batch, model, dataset) do
    # Estimate learning stage based on model complexity
    stage = cond do
      length(model.meta_rules) >= 3 -> 7  # Ecosystem stage
      length(model.meta_rules) >= 2 -> 6  # Meta stage
      length(model.parameters) >= 20 -> 4  # Network stage
      length(model.parameters) >= 10 -> 3  # Branch stage
      true -> 2  # Sprout stage
    end

    # Add stage to batch statistics
    if is_map(batch) && Map.has_key?(batch, :statistics) do
      stats = Map.put(batch.statistics, :learning_stage, stage)
      stats = Map.put(stats, :parameter_count, length(model.parameters))
      stats = Map.put(stats, :meta_rule_count, length(model.meta_rules))
      %{batch | statistics: stats}
    else
      batch
    end
  end

  # Deep comparison for parameters
  def parameters_changed?(old_params, new_params) do
    # Compare parameter values with tolerance for floating point
    param_pairs = Enum.zip(old_params, new_params)

    Enum.any?(param_pairs, fn {old, new} ->
      # Check if name matches
      if old[:name] == new[:name] do
        # Compare values with some tolerance for floating point
        abs(old[:value] - new[:value]) > 0.0001
      else
        # Different parameters (should not happen)
        true
      end
    end)
  end

  # Apply boosted learning to ensure progress
  def apply_boosted_learning(parameters, heuristics, batch) do
    # Get important feature names for each category from batch
    key_features = extract_key_features_from_batch(batch)

    # Boost learning for key features
    {updated_parameters, updated_batch} =
      Enum.reduce(key_features, {parameters, batch}, fn feature_name, {params, current_batch} ->
        # Find parameter matching this feature
        param_idx = Enum.find_index(params, fn p -> p[:name] == feature_name end)

        if param_idx do
          param = Enum.at(params, param_idx)
          # Apply a small boost to ensure learning progresses
          new_value = param[:value] + 0.01
          new_param = %{param | value: new_value}
          new_params = List.replace_at(params, param_idx, new_param)
          {new_params, current_batch}
        else
          {params, current_batch}
        end
      end)

    {updated_parameters, updated_batch}
  end

  # Extract key features from batch
  def extract_key_features_from_batch(batch) do
    # Get data points
    data_points = Heuristics.extract_data_points(batch)

    # Extract categories
    categories =
      Enum.map(data_points, fn dp ->
        label = dp[:label]
        if is_atom(label), do: Atom.to_string(label), else: label
      end)
      |> Enum.uniq()

    # Get config
    config =
      if is_map(batch) && Map.has_key?(batch, :statistics) &&
         is_map(batch.statistics) && Map.has_key?(batch.statistics, :config) do
        batch.statistics.config
      else
        %{key_terms: %{}}
      end

    # Get key terms for all categories
    Enum.flat_map(categories, fn category ->
      case Map.get(config, :key_terms, %{}) do
        key_terms when is_map(key_terms) ->
          Map.get(key_terms, category, [])
        _ ->
          []
      end
    end)
    |> Enum.uniq()
  end

  # Merge batch statistics back into the dataset
  def merge_batch_statistics(dataset, batches) do
    fold dataset do
      case(dataset(data_points, statistics)) ->
        # Combine all batch statistics into the dataset
        combined_stats =
          Enum.reduce(batches, statistics, fn batch, acc ->
            if is_map(batch) && Map.has_key?(batch, :statistics) do
              Map.merge(acc, batch.statistics, fn _k, v1, v2 ->
                # For conflicting keys, prefer the more recent value
                if is_map(v2) && Map.has_key?(v2, :update_time), do: v2, else: v1
              end)
            else
              acc
            end
          end)

        Learning.dataset(data_points, combined_stats)
    end
  end
end
