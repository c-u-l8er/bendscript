defmodule TextClassifier do
  import KernelShtf.BenBen
  alias KernelShtf.MetaLearning
  alias KernelShtf.MetaLearning.Types.Learning

  # Define basic types using the bend/phrenia system
  phrenia TextData do
    document(text, features, category)
    classifier(model, vocabulary, categories)
    feature(name, weight)
  end

  @doc """
  Creates a new text classifier with meta-learning capabilities.
  The classifier adjusts its feature weights and selection strategy
  automatically based on the data it encounters.
  """
  def new(initial_vocabulary, categories) do
    # Create a parameter for each word in the vocabulary
    parameters =
      Enum.map(initial_vocabulary, fn word ->
        MetaLearning.new_parameter(word, 0.1, 0.05)
      end)

    # Basic update heuristics - how words affect classification
    heuristics = [
      # Increase weights for words that appear in their assigned category
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(initial_vocabulary, param_name) end,
        fn name, value, _rate, features, label, _batch ->
          # If word appears in text and the label matches, increase weight
          if Map.get(features, name, 0) > 0 && features.category == label do
            0.1
          else
            0.0
          end
        end,
        0.7
      ),

      # Decrease weights for words that appear in wrong categories
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(initial_vocabulary, param_name) end,
        fn name, value, _rate, features, label, _batch ->
          # If word appears but in wrong category, decrease weight
          if Map.get(features, name, 0) > 0 && features.category != label do
            -0.1
          else
            0.0
          end
        end,
        0.5
      ),

      # Decay unused words
      MetaLearning.new_heuristic(
        fn param_name -> Enum.member?(initial_vocabulary, param_name) end,
        fn name, value, _rate, features, _label, _batch ->
          if Map.get(features, name, 0) == 0 do
            -0.01
          else
            0.0
          end
        end,
        0.3
      )
    ]

    # Meta-rules that modify how the classifier learns
    meta_rules = [
      # Increase learning rate for features that consistently correlate with categories
      MetaLearning.new_meta_rule(
        %{param_name: :any, expected_pattern: :convergence},
        fn heuristics, dataset ->
          # Find heuristics that are working well
          effective_heuristics =
            Enum.map(heuristics, fn
              %{variant: :heuristic, condition: cond, action: action, confidence: conf}
              when conf < 0.9 ->
                # Increase confidence for effective heuristics
                Learning.heuristic(cond, action, conf + 0.05)

              other ->
                other
            end)

          {effective_heuristics, dataset}
        end,
        0.7
      ),

      # Reduce oscillation by adding momentum
      MetaLearning.new_meta_rule(
        %{param_name: :any, expected_pattern: :oscillation},
        fn heuristics, dataset ->
          # Add momentum to oscillating parameters
          updated_heuristics =
            Enum.map(heuristics, fn
              %{variant: :heuristic, condition: cond, action: action, confidence: conf} ->
                # Create a damped version of the action
                damped_action = fn name, value, rate, features, label, batch ->
                  # Get original update
                  raw_update = action.(name, value, rate, features, label, batch)

                  # Get previous updates from dataset statistics
                  previous_updates =
                    fold dataset do
                      case(dataset(data_points, statistics)) ->
                        Map.get(statistics, {:param_updates, name}, [])
                    end

                  # Apply momentum (weighted average with previous updates)
                  if previous_updates == [] do
                    raw_update
                  else
                    avg_prev = Enum.sum(previous_updates) / length(previous_updates)
                    raw_update * 0.7 + avg_prev * 0.3
                  end
                end

                Learning.heuristic(cond, damped_action, conf)

              other ->
                other
            end)

          {updated_heuristics, dataset}
        end,
        0.8
      ),

      # Add feature pruning for irrelevant words
      MetaLearning.new_meta_rule(
        %{heuristic_type: :feature_pruning},
        fn heuristics, dataset ->
          # Create a new heuristic that removes features with very low weights
          pruning_heuristic =
            Learning.heuristic(
              %{type: :feature_pruning},
              fn _name, value, _rate, _features, _label, _batch ->
                if abs(value) < 0.01 do
                  # Further push toward zero
                  -value * 0.5
                else
                  0.0
                end
              end,
              0.4
            )

          {[pruning_heuristic | heuristics], dataset}
        end,
        0.5
      )
    ]

    # Create model
    model = MetaLearning.new_model(parameters, heuristics, meta_rules)

    # Return classifier
    TextData.classifier(model, initial_vocabulary, categories)
  end

  @doc """
  Processes a document, extracting features for the classifier
  """
  def extract_features(document, vocabulary) do
    # Tokenize text
    words =
      document
      |> String.downcase()
      |> String.replace(~r/[^\w\s]/, "")
      |> String.split(~r/\s+/)

    # Count occurrences of vocabulary words
    features =
      Enum.reduce(vocabulary, %{}, fn word, acc ->
        count = Enum.count(words, &(&1 == word))
        Map.put(acc, word, count)
      end)

    features
  end

  @doc """
  Train the classifier on a set of documents
  """
  def train(classifier, documents, epochs \\ 5) do
    # Manually extract the necessary parts from the classifier
    model = classifier.model
    vocabulary = classifier.vocabulary
    categories = classifier.categories

    # Convert documents to dataset
    dataset = prepare_dataset(documents, vocabulary)

    # Train using the meta-learning system
    trained_model = MetaLearning.train(model, dataset, epochs)

    # Create a new classifier with the trained model
    TextData.classifier(trained_model, vocabulary, categories)
  end

  @doc """
  Prepare training dataset from documents
  """
  def prepare_dataset(documents, vocabulary) do
    # Convert documents to data points
    data_points =
      Enum.map(documents, fn doc ->
        fold doc do
          case(document(text, features, category)) ->
            # Extract features
            features = extract_features(text, vocabulary)

            # Add category to features for the heuristics
            features_with_category = Map.put(features, :category, category)

            # Create data point
            MetaLearning.new_data_point(features_with_category, category)
        end
      end)

    # Create dataset with initial statistics
    MetaLearning.new_dataset(data_points, %{
      size: length(data_points),
      categories: Enum.map(data_points, fn dp -> dp.label end) |> Enum.uniq() |> length(),
      avg_features:
        (Enum.map(data_points, fn dp -> map_size(dp.features) end)
         |> Enum.sum()) / length(data_points)
    })
  end

  @doc """
  Classify a new document
  """
  def classify(classifier, text) do
    # Extract model, vocabulary and categories directly using map access
    model = classifier.model
    vocabulary = classifier.vocabulary
    categories = classifier.categories

    # Extract features
    features = extract_features(text, vocabulary)

    # Get parameters (word weights)
    parameters = model.parameters

    # Calculate score for each category
    scores =
      Enum.map(categories, fn category ->
        # Sum weights of words that appear in this document
        score =
          Enum.reduce(parameters, 0, fn param, acc ->
            # Use the parameter's name and value directly
            name = param.name
            value = param.value

            # Multiply word count by its weight
            acc + Map.get(features, name, 0) * value
          end)

        {category, score}
      end)

    # Return category with highest score
    Enum.max_by(scores, fn {_category, score} -> score end)
  end

  @doc """
  Extract the most important words for each category
  """
  def important_features(classifier) do
    # Get parameters (word weights) directly
    parameters = classifier.model.parameters

    # Get word weights using direct map access
    word_weights =
      parameters
      |> Enum.map(fn param ->
        {param.name, param.value}
      end)
      |> Enum.filter(fn {_name, value} -> abs(value) > 0.1 end)
      |> Enum.sort_by(fn {_name, value} -> -abs(value) end)

    # Return top words
    Enum.take(word_weights, 10)
  end
end
