defmodule HelloNxTest do
  use ExUnit.Case
  doctest HelloNx

  test "greets the world" do
    assert HelloNx.hello() == :world
  end
end
