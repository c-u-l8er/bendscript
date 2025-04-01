defmodule EmergenceTestHelper do
  @moduledoc """
  Helper functions for the meta-learning emergence tests
  """

  # Extract parameter values as a map
  def extract_parameter_values(parameters) do
    Enum.map(parameters, fn
      %{variant: :parameter, name: name, value: value} -> {name, value}
      parameter -> {parameter[:name], parameter[:value]}
    end)
    |> Enum.into(%{})
  end

  # Copy learned weights from one classifier to another
  def copy_learned_weights(target_classifier, source_classifier) do
    # Extract source weights
    source_weights = extract_parameter_values(source_classifier.model.parameters)

    # Update target parameters with source weights when available
    updated_parameters = Enum.map(target_classifier.model.parameters, fn param ->
      name = param[:name]
      if Map.has_key?(source_weights, name) do
        %{param | value: source_weights[name]}
      else
        param
      end
    end)

    # Return updated classifier
    %{target_classifier | model: %{target_classifier.model | parameters: updated_parameters}}
  end

  # Extract learned weights as a map
  def extract_learned_weights(classifier) do
    extract_parameter_values(classifier.model.parameters)
  end
end
