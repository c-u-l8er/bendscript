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
    IO.puts("DEBUG: Starting learn_epoch")
    # Extract batches from dataset
    batches = Batching.create_batches(dataset)
    IO.puts("DEBUG: Created #{length(batches)} batches")

    # Learn from each batch sequentially
    {final_model, updated_batches} =
      Enum.reduce(batches, {model, []}, fn batch, {current_model, processed_batches} ->
        # Direct map access for model structure
        parameters = current_model.parameters
        heuristics = current_model.heuristics
        meta_rules = current_model.meta_rules

        IO.puts("DEBUG: Processing batch with #{length(batch[:data_subset] || [])} data points")

        # Apply heuristics to determine how to update parameters
        {updated_parameters, updated_batch} =
          Heuristics.apply_heuristics(parameters, heuristics, batch)

        # Check if parameters were actually updated
        param_changed = parameters != updated_parameters
        IO.puts("DEBUG: Parameters changed: #{param_changed}")

        # Track learning statistics to update dataset
        updated_batch =
          Batching.update_statistics(updated_batch, parameters, updated_parameters)

        # Save the updated batch
        {%{current_model | parameters: updated_parameters},
         [updated_batch | processed_batches]}
      end)

    # Merge batch statistics back into the dataset
    updated_dataset = merge_batch_statistics(dataset, updated_batches)

    {final_model, updated_dataset}
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
