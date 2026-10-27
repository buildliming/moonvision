///|
test "nearest mask resize resolves exact center ties on both axes" {
  for channels in [1, 3, 4] {
    let input : Sample = {
      image: {
        width: 2,
        height: 2,
        channels,
        data: Array::makei(4 * channels, i => (i / channels * 64).to_byte()),
      },
      mask: Some([0, 17, 2147483647, 42]),
      boxes: [],
    }
    let result = augment(
      input,
      [{ transform: Resize(49, 49), probability: 1.0, }],
      0,
      "mask-center-tie",
    )
    let expected = Array::makei(49 * 49, i => {
      let x = i % 49
      let y = i / 49
      if y < 24 {
        if x < 24 {
          0
        } else {
          17
        }
      } else if x < 24 {
        2147483647
      } else {
        42
      }
    })
    guard result.sample.mask is Some(mask) else {
      fail("mask must be retained")
    }
    for i in 0..<expected.length() {
      assert_eq(mask[i], expected[i])
    }
    assert_eq(replay(input, result.replay), result.sample)
    assert_eq(input.mask, Some([0, 17, 2147483647, 42]))
  }
}

///|
test "nearest mask indices stay exact at the typed pixel budget" {
  assert_eq(
    nearest_source_index(MAX_PIXELS - 1, MAX_PIXELS, MAX_PIXELS),
    MAX_PIXELS - 1,
  )
  assert_eq(nearest_source_index(MAX_PIXELS - 1, 1, MAX_PIXELS), 0)
  assert_eq(nearest_source_index(0, MAX_PIXELS, 1), MAX_PIXELS / 2)
  assert_eq(nearest_source_index(8388607, 2, MAX_PIXELS - 1), 1)
}

///|
test "nearest mask resize matches integer pixel-center intervals" {
  for input_size in 1..<=32 {
    let input : Sample = {
      image: {
        width: input_size,
        height: 1,
        channels: 1,
        data: Array::make(input_size, b'\x00'),
      },
      mask: Some(Array::makei(input_size, i => i)),
      boxes: [],
    }
    for output_size in 1..<=96 {
      let output = run_transform(input, Resize(output_size, 1))
      guard output.mask is Some(mask) else { fail("mask must be retained") }
      for destination in 0..<output_size {
        let source = mask[destination]
        let center = (2 * destination + 1) * input_size
        assert_true(2 * source * output_size <= center)
        assert_true(center < 2 * (source + 1) * output_size)
      }
    }
  }
}
