# Compile

With `hdf5` library, `xmake` (install by `curl -fsSL https://xmake.io/shget.text | bash`) installed,
compile the program by `xmake build mprc`. Then, you can use the program `mprc` under
`./build/<your OS name>/<architecture>/release/`, which you can copy to some directory in your
`$PATH`.

Note: you may need to edit the link path of the `hdf5` library in the line of `xmake.lua`, which
begins with `add_ldflags`.

# Usage

See `mprc --help`.
