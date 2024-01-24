defmodule LinearRegression do
  import Nx.Defn

  def demo() do
    target_m = :rand.normal(0.0, 10.0)
    target_b = :rand.normal(0.0, 5.0)
    target_fn = fn x -> target_m * x + target_b end

    data =
      Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, target_fn)) end)

    IO.puts("Target m: #{target_m}\tTarget b: #{target_b}\n")
    {time, {m, b}} = :timer.tc(__MODULE__, :train, [100, data])
    IO.puts("Learned m: #{to_scalar(m)}\tLearned b: #{to_scalar(b)}")
    IO.puts("Training time: #{time / 1_000_000}s")
  end

  defn add_two(a, b) do
    a + b
  end

  defn predict({m, b}, x) do
    m * x + b
  end

  # mean-squared error (MSE)
  defn loss(params, x, y) do
    y_pred = predict(params, x)
    Nx.mean(Nx.pow(y - y_pred, 2))
  end

  #

  defn update({m, b} = params, inp, tar) do
    {grad_m, grad_b} = grad(params, &loss(&1, inp, tar))
    {m - grad_m * 0.01, b - grad_b * 0.01}
  end

  defn init_random_params do
    m = Nx.random_normal({}, 0.0, 0.1)
    b = Nx.random_normal({}, 0.0, 0.1)
    {m, b}
  end

  def train(epochs, data) do
    init_params = init_random_params()

    for _ <- 1..epochs, reduce: init_params do
      acc ->
        data
        |> Enum.take(200)
        |> Enum.reduce(
          acc,
          fn batch, cur_params ->
            {inp, tar} = Enum.unzip(batch)
            x = Nx.tensor(inp)
            y = Nx.tensor(tar)
            update(cur_params, x, y)
          end
        )
    end
  end

  # defn init_random_params do
  #   m = random_normal(Nx.tensor([0.0]), Nx.tensor([[0.1]]))
  #   b = random_normal(Nx.tensor([0.0]), Nx.tensor([[0.1]]))
  #   {m, b}
  # end

  defn random_normal(mu, sigma) do
    key = Enum.random(1..100_000)

    {a, _} =
      Nx.Random.multivariate_normal(Nx.Random.key(key), mu, sigma)

    Nx.reshape(a, {})
  end

  # def to_scalar(t) do
  #   t
  #   |> Nx.reshape({})
  #   |> Nx.to_number()
  # end

  # https://github.com/elixir-nx/nx/blob/main/exla/examples/regression.exs#L69-L72
  def to_scalar(t) do
    t
    |> Nx.squeeze()
    |> Nx.backend_transfer()
    |> Nx.to_number()
  end
end
