defmodule KernelShtf.MetaLearning.Batching do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning.Types.Learning

  # Create batches from a dataset for mini-batch learning
  def create_batches(dataset) do
    fold dataset do
      case(dataset(data_points, statistics)) ->
        # Determine appropriate batch size based on dataset statistics
        batch_size = determine_batch_size(statistics)

        # Group data points into batches
        data_points
        |> Enum.chunk_every(batch_size)
        |> Enum.map(fn chunk ->
          # Use the 3-parameter version with empty statistics
          Learning.batch(chunk, batch_size, %{})
        end)
    end
  end

  # Determine optimal batch size based on dataset statistics
  def determine_batch_size(statistics) do
    # Default batch size if no statistics available
    default_size = 32

    # Extract data size from statistics if available
    data_size = Map.get(statistics, :size, 1000)

    cond do
      data_size < 100 -> max(1, div(data_size, 4))
      data_size < 1000 -> max(8, div(data_size, 10))
      data_size < 10000 -> max(16, div(data_size, 20))
      true -> default_size
    end
  end

  # Update batch statistics based on parameter updates
  def update_statistics(batch, old_parameters, new_parameters) do
    # Check if batch has the proper structure
    if batch[:variant] == :batch do
      # Calculate parameter changes
      param_changes = calculate_parameter_changes(old_parameters, new_parameters)

      # Update batch statistics with these changes
      data_subset = batch[:data_subset]
      batch_size = batch[:batch_size]

      # Update batch statistics with these changes
      updated_stats = batch_stats_update(batch, param_changes)

      Learning.batch(data_subset, batch_size, updated_stats)
    else
      # Return the batch unchanged if it doesn't have the expected structure
      batch
    end
  end

  # Calculate changes between old and new parameters
  def calculate_parameter_changes(old_params, new_params) do
    # Zip parameters together and calculate differences
    Enum.zip(old_params, new_params)
    |> Enum.map(fn {old, new} ->
      # Handle direct map access instead of fold pattern matching
      if old[:variant] == :parameter && new[:variant] == :parameter &&
           old[:name] == new[:name] do
        {old[:name], new[:value] - old[:value]}
      else
        # Default case for unmatched parameters
        {nil, 0.0}
      end
    end)
    # Remove any nil entries
    |> Enum.reject(fn {name, _} -> is_nil(name) end)
    |> Enum.into(%{})
  end

  # Update batch statistics with parameter changes
  def batch_stats_update(batch, param_changes) do
    # Get existing statistics or initialize new ones
    existing_stats = Map.get(batch, :statistics, %{})

    # Update statistics with parameter changes
    updated_stats =
      Map.merge(existing_stats, %{
        last_update: param_changes,
        update_time: DateTime.utc_now(),
        update_count: Map.get(existing_stats, :update_count, 0) + 1
      })

    updated_stats
  end

  def batch_to_dataset(batch) do
    if is_map(batch) && Map.has_key?(batch, :variant) && batch.variant == :batch do
      Learning.dataset(
        batch.data_subset,
        Map.get(batch, :statistics, %{})
      )
    else
      # Return a default dataset if input is invalid
      Learning.dataset([], %{})
    end
  end
end
