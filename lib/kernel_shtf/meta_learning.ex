defmodule KernelShtf.MetaLearning do
  @moduledoc """
  A meta-learning system that can modify its own learning algorithms.

  This system uses recursive data structures to represent learning models,
  and applies meta-rules to evolve those models based on observed patterns
  and effectiveness.
  """

  # Re-export the public API
  defdelegate train(initial_model, dataset, epochs), to: KernelShtf.MetaLearning.Core

  # Helper functions to create model components
  alias KernelShtf.MetaLearning.Types.Learning

  @doc """
  Creates a new learning model with the given parameters, heuristics, and meta-rules.
  """
  def new_model(parameters, heuristics, meta_rules) do
    Learning.model(parameters, heuristics, meta_rules)
  end

  @doc """
  Creates a new parameter with the given name, initial value, and update rate.
  """
  def new_parameter(name, value, update_rate \\ 0.1) do
    Learning.parameter(name, value, update_rate)
  end

  @doc """
  Creates a new heuristic with the given condition, action, and confidence.
  """
  def new_heuristic(condition, action, confidence \\ 0.5) do
    Learning.heuristic(condition, action, confidence)
  end

  @doc """
  Creates a new meta-rule with the given pattern, transformation, and priority.
  """
  def new_meta_rule(pattern, transformation, priority \\ 0.5) do
    Learning.metaRule(pattern, transformation, priority)
  end

  @doc """
  Creates a new dataset with the given data points and optional statistics.
  """
  def new_dataset(data_points, statistics \\ %{}) do
    Learning.dataset(data_points, statistics)
  end

  @doc """
  Creates a new data point with the given features and label.
  """
  def new_data_point(features, label) do
    Learning.dataPoint(features, label)
  end
end
