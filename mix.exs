defmodule BendScript.MixProject do
  use Mix.Project

  def project do
    [
      app: :bendscript,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      # Add the following to enable distributed tests
      elixirc_options: [
        {:warnings_as_errors, false}
      ],
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        test: :test
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:uuid, "~> 1.1"},
      {:memento, "~> 0.3.2"},
      {:poolboy, "~> 1.5"},
      {:gen_stage, "~> 1.2"},
      {:flow, "~> 1.2"},
      {:broadway, "~> 1.0"}
    ]
  end
end
