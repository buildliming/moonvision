// Generated from independent NumPy/PyTorch geometry references.

///|
fn extended_sample_close(
  actual : @moonaug.Sample,
  expected : @moonaug.Sample,
  tolerance : Int,
) -> Unit raise {
  assert_eq(
    (actual.image.width, actual.image.height, actual.image.channels),
    (expected.image.width, expected.image.height, expected.image.channels),
  )
  assert_eq(actual.image.data.length(), expected.image.data.length())
  for i = 0; i < actual.image.data.length(); i = i + 1 {
    let delta = actual.image.data[i].to_int() - expected.image.data[i].to_int()
    assert_true(delta >= -tolerance && delta <= tolerance)
  }
  assert_eq(actual.mask, expected.mask)
  assert_eq(actual.boxes.length(), expected.boxes.length())
  for i = 0; i < actual.boxes.length(); i = i + 1 {
    let a = actual.boxes[i]
    let b = expected.boxes[i]
    assert_true(a.x_min < a.x_max && a.y_min < a.y_max)
    assert_eq(a.label, b.label)
    for
      delta in [
        a.x_min - b.x_min,
        a.y_min - b.y_min,
        a.x_max - b.x_max,
        a.y_max - b.y_max,
      ] {
      assert_true(delta >= -1.0e-10 && delta <= 1.0e-10)
    }
  }
}

///|
test "extended independent geometry: asymmetric_per_channel_pad" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Pad(@moonaug.Padding::{
          left: 1,
          top: 2,
          right: 2,
          bottom: 0,
          fill: [7, 29, 131],
          mask_value: 1000,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 7,
        height: 5,
        channels: 3,
        data: [
          7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131,
          7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131,
          7, 29, 131, 7, 29, 131, 7, 29, 131, 5, 12, 19, 26, 33, 40, 47, 54, 61,
          68, 75, 82, 7, 29, 131, 7, 29, 131, 7, 29, 131, 89, 96, 103, 110, 117,
          124, 131, 138, 145, 152, 159, 166, 7, 29, 131, 7, 29, 131, 7, 29, 131,
          173, 180, 187, 194, 201, 208, 215, 222, 229, 236, 243, 250, 7, 29, 131,
          7, 29, 131,
        ],
      },
      mask: Some([
        1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
        1000, 1000, 0, 0, 1, 1, 1000, 1000, 1000, 2, 3, 3, 1, 1000, 1000, 1000, 2,
        2, 4, 4, 1000, 1000,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 1.25,
          y_min: 2.5,
          x_max: 4.6,
          y_max: 4.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 1.0,
          y_min: 2.0,
          x_max: 2.0,
          y_max: 3.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 4.0,
          y_min: 4.0,
          x_max: 5.0,
          y_max: 5.0,
          label: "edge",
        },
      ],
    },
    0,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: zero_extent_pad" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Pad(@moonaug.Padding::{
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          fill: [0, 0, 0],
          mask_value: 9,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 4,
        height: 3,
        channels: 3,
        data: [
          5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
          131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
          236, 243, 250,
        ],
      },
      mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.25,
          y_min: 0.5,
          x_max: 3.6,
          y_max: 2.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 1.0,
          y_max: 1.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 3.0,
          y_min: 2.0,
          x_max: 4.0,
          y_max: 3.0,
          label: "edge",
        },
      ],
    },
    0,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: right_bottom_pad" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Pad(@moonaug.Padding::{
          left: 0,
          top: 0,
          right: 1,
          bottom: 2,
          fill: [255, 11, 0],
          mask_value: 2147483647,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 5,
        height: 5,
        channels: 3,
        data: [
          5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 255, 11, 0, 89, 96, 103,
          110, 117, 124, 131, 138, 145, 152, 159, 166, 255, 11, 0, 173, 180, 187,
          194, 201, 208, 215, 222, 229, 236, 243, 250, 255, 11, 0, 255, 11, 0, 255,
          11, 0, 255, 11, 0, 255, 11, 0, 255, 11, 0, 255, 11, 0, 255, 11, 0, 255,
          11, 0, 255, 11, 0, 255, 11, 0,
        ],
      },
      mask: Some([
        0, 0, 1, 1, 2147483647, 2, 3, 3, 1, 2147483647, 2, 2, 4, 4, 2147483647, 2147483647,
        2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647, 2147483647,
        2147483647, 2147483647,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.25,
          y_min: 0.5,
          x_max: 3.6,
          y_max: 2.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 1.0,
          y_max: 1.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 3.0,
          y_min: 2.0,
          x_max: 4.0,
          y_max: 3.0,
          label: "edge",
        },
      ],
    },
    0,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: letterbox_7x7_True" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
          width: 7,
          height: 7,
          fill: [23, 47, 91],
          mask_value: 1024,
          scale_up: true,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 7,
        height: 7,
        channels: 3,
        data: [
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          23, 47, 91, 5, 12, 19, 13, 20, 27, 24, 31, 39, 37, 44, 51, 48, 55, 63,
          61, 68, 75, 68, 75, 82, 39, 46, 53, 46, 53, 60, 58, 65, 72, 70, 77, 84,
          82, 89, 96, 94, 101, 108, 102, 109, 116, 89, 96, 103, 97, 104, 111, 109,
          116, 123, 121, 128, 135, 133, 139, 146, 145, 152, 159, 152, 159, 166, 139,
          146, 153, 147, 154, 161, 159, 166, 173, 171, 178, 185, 183, 190, 197, 195,
          202, 209, 202, 209, 216, 173, 180, 187, 180, 187, 195, 193, 200, 207, 205,
          211, 219, 217, 224, 231, 229, 236, 243, 236, 243, 250, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
        ],
      },
      mask: Some([
        1024, 1024, 1024, 1024, 1024, 1024, 1024, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 1024,
        1024, 1024, 1024, 1024, 1024, 1024,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.4375,
          y_min: 1.8333333333333335,
          x_max: 6.3,
          y_max: 5.333333333333334,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 1.0,
          x_max: 1.75,
          y_max: 2.666666666666667,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 5.25,
          y_min: 4.333333333333334,
          x_max: 7.0,
          y_max: 6.0,
          label: "edge",
        },
      ],
    },
    1,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: letterbox_3x7_True" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
          width: 3,
          height: 7,
          fill: [23, 47, 91],
          mask_value: 1024,
          scale_up: true,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 3,
        height: 7,
        channels: 3,
        data: [
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          30, 37, 44, 58, 65, 72, 86, 93, 100, 156, 163, 170, 184, 191, 198, 212,
          219, 226, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
        ],
      },
      mask: Some([
        1024, 1024, 1024, 1024, 1024, 1024, 0, 1, 1, 2, 4, 4, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 1024, 1024,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.1875,
          y_min: 2.3333333333333335,
          x_max: 2.7,
          y_max: 3.7333333333333334,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 2.0,
          x_max: 0.75,
          y_max: 2.6666666666666665,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 2.25,
          y_min: 3.333333333333333,
          x_max: 3.0,
          y_max: 4.0,
          label: "edge",
        },
      ],
    },
    1,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: letterbox_9x7_True" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
          width: 9,
          height: 7,
          fill: [23, 47, 91],
          mask_value: 1024,
          scale_up: true,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 9,
        height: 7,
        channels: 3,
        data: [
          5, 12, 19, 9, 16, 23, 18, 25, 32, 27, 34, 41, 37, 44, 51, 46, 53, 60, 55,
          62, 69, 64, 72, 79, 68, 75, 82, 17, 24, 31, 20, 27, 34, 30, 37, 44, 39,
          46, 53, 49, 56, 63, 58, 65, 72, 67, 74, 81, 76, 83, 91, 80, 87, 94, 53,
          60, 67, 57, 63, 71, 66, 73, 80, 75, 82, 89, 85, 92, 99, 94, 101, 108, 103,
          110, 117, 113, 119, 127, 116, 123, 130, 89, 96, 103, 93, 100, 107, 102,
          109, 116, 111, 118, 125, 121, 128, 135, 130, 137, 144, 139, 146, 153, 149,
          156, 163, 152, 159, 166, 125, 132, 139, 129, 136, 142, 138, 145, 152, 147,
          154, 161, 156, 164, 171, 166, 173, 180, 175, 182, 189, 185, 191, 198, 188,
          195, 202, 161, 168, 175, 165, 172, 179, 174, 181, 188, 183, 190, 197, 193,
          200, 207, 202, 209, 216, 211, 218, 225, 221, 228, 235, 224, 231, 238, 173,
          180, 187, 177, 183, 190, 186, 193, 200, 195, 202, 209, 205, 212, 219, 214,
          221, 228, 223, 230, 237, 233, 239, 247, 236, 243, 250,
        ],
      },
      mask: Some([
        0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3,
        1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 4,
        4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 4,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.5625,
          y_min: 1.1666666666666667,
          x_max: 8.1,
          y_max: 6.066666666666667,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 2.25,
          y_max: 2.3333333333333335,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 6.75,
          y_min: 4.666666666666667,
          x_max: 9.0,
          y_max: 7.0,
          label: "edge",
        },
      ],
    },
    1,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: letterbox_9x9_False" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
          width: 9,
          height: 9,
          fill: [23, 47, 91],
          mask_value: 1024,
          scale_up: false,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 9,
        height: 9,
        channels: 3,
        data: [
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 5, 12, 19,
          26, 33, 40, 47, 54, 61, 68, 75, 82, 23, 47, 91, 23, 47, 91, 23, 47, 91,
          23, 47, 91, 23, 47, 91, 89, 96, 103, 110, 117, 124, 131, 138, 145, 152,
          159, 166, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 173,
          180, 187, 194, 201, 208, 215, 222, 229, 236, 243, 250, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23,
          47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91, 23, 47, 91,
        ],
      },
      mask: Some([
        1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 0, 0, 1, 1, 1024, 1024, 1024, 1024, 1024, 2, 3, 3, 1, 1024,
        1024, 1024, 1024, 1024, 2, 2, 4, 4, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 2.25,
          y_min: 3.5,
          x_max: 5.6,
          y_max: 5.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 2.0,
          y_min: 3.0,
          x_max: 3.0,
          y_max: 4.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 5.0,
          y_min: 5.0,
          x_max: 6.0,
          y_max: 6.0,
          label: "edge",
        },
      ],
    },
    1,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: letterbox_1x7_True" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [
      @moonaug.Step::{
        transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
          width: 1,
          height: 7,
          fill: [23, 47, 91],
          mask_value: 1024,
          scale_up: true,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 1,
        height: 7,
        channels: 3,
        data: [
          23, 47, 91, 23, 47, 91, 23, 47, 91, 121, 128, 135, 23, 47, 91, 23, 47,
          91, 23, 47, 91,
        ],
      },
      mask: Some([1024, 1024, 1024, 3, 1024, 1024, 1024]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.0625,
          y_min: 3.1666666666666665,
          x_max: 0.9,
          y_max: 3.8666666666666667,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 3.0,
          x_max: 0.25,
          y_max: 3.3333333333333335,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 0.75,
          y_min: 3.6666666666666665,
          x_max: 1.0,
          y_max: 4.0,
          label: "edge",
        },
      ],
    },
    1,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
test "extended independent geometry: positive_box_area_must_not_underflow" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{ width: 2, height: 1, channels: 1, data: [10, 20], },
    mask: Some([1, 2]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0e-200,
        y_max: 1.0e-200,
        label: "tiny",
      },
    ],
  }
  let result = @moonaug.augment(
    input,
    [@moonaug.Step::{ transform: @moonaug.Crop(0, 0, 2, 1), probability: 1.0, }],
    20261026,
    "extended-reference",
  )
  extended_sample_close(
    result.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 2,
        height: 1,
        channels: 1,
        data: [10, 20],
      },
      mask: Some([1, 2]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 1.0e-200,
          y_max: 1.0e-200,
          label: "tiny",
        },
      ],
    },
    0,
  )
  assert_eq(input, @moonaug.Sample::{
    image: @moonaug.Image::{ width: 2, height: 1, channels: 1, data: [10, 20], },
    mask: Some([1, 2]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0e-200,
        y_max: 1.0e-200,
        label: "tiny",
      },
    ],
  })
  let record = @moonaug.replay_from_json(@moonaug.replay_to_json(result.replay))
  assert_eq(@moonaug.replay(input, record), result.sample)
}

///|
fn extended_crop_expected(
  x : Int,
  y : Int,
  width : Int,
  height : Int,
) -> @moonaug.Sample raise {
  match (x, y, width, height) {
    (0, 0, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [5, 12, 19, 5, 12, 19, 5, 12, 19, 5, 12, 19],
        },
        mask: Some([0, 0, 0, 0]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.5,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [89, 96, 103, 89, 96, 103, 89, 96, 103, 89, 96, 103],
        },
        mask: Some([2, 2, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.5,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (0, 2, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [173, 180, 187, 173, 180, 187, 173, 180, 187, 173, 180, 187],
        },
        mask: Some([2, 2, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.5,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.2000000000000002,
            label: "main",
          },
        ],
      }
    (1, 0, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [26, 33, 40, 26, 33, 40, 26, 33, 40, 26, 33, 40],
        },
        mask: Some([0, 0, 0, 0]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 1, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [110, 117, 124, 110, 117, 124, 110, 117, 124, 110, 117, 124],
        },
        mask: Some([3, 3, 3, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 2, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [194, 201, 208, 194, 201, 208, 194, 201, 208, 194, 201, 208],
        },
        mask: Some([2, 2, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.2000000000000002,
            label: "main",
          },
        ],
      }
    (2, 0, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [47, 54, 61, 47, 54, 61, 47, 54, 61, 47, 54, 61],
        },
        mask: Some([1, 1, 1, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (2, 1, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [131, 138, 145, 131, 138, 145, 131, 138, 145, 131, 138, 145],
        },
        mask: Some([3, 3, 3, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (2, 2, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [215, 222, 229, 215, 222, 229, 215, 222, 229, 215, 222, 229],
        },
        mask: Some([4, 4, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.2000000000000002,
            label: "main",
          },
        ],
      }
    (3, 0, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [68, 75, 82, 68, 75, 82, 68, 75, 82, 68, 75, 82],
        },
        mask: Some([1, 1, 1, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 1.2000000000000002,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (3, 1, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [152, 159, 166, 152, 159, 166, 152, 159, 166, 152, 159, 166],
        },
        mask: Some([1, 1, 1, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.2000000000000002,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (3, 2, 1, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [236, 243, 250, 236, 243, 250, 236, 243, 250, 236, 243, 250],
        },
        mask: Some([4, 4, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.2000000000000002,
            y_max: 1.2000000000000002,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [5, 12, 19, 5, 12, 19, 89, 96, 103, 89, 96, 103],
        },
        mask: Some([0, 0, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.5,
            y_min: 0.5,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [89, 96, 103, 89, 96, 103, 173, 180, 187, 173, 180, 187],
        },
        mask: Some([2, 2, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.5,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.6,
            label: "main",
          },
        ],
      }
    (1, 0, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [26, 33, 40, 26, 33, 40, 110, 117, 124, 110, 117, 124],
        },
        mask: Some([0, 0, 3, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.5,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 1, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [110, 117, 124, 110, 117, 124, 194, 201, 208, 194, 201, 208],
        },
        mask: Some([3, 3, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.6,
            label: "main",
          },
        ],
      }
    (2, 0, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [47, 54, 61, 47, 54, 61, 131, 138, 145, 131, 138, 145],
        },
        mask: Some([1, 1, 3, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.5,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (2, 1, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [131, 138, 145, 131, 138, 145, 215, 222, 229, 215, 222, 229],
        },
        mask: Some([3, 3, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.6,
            label: "main",
          },
        ],
      }
    (3, 0, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [68, 75, 82, 68, 75, 82, 152, 159, 166, 152, 159, 166],
        },
        mask: Some([1, 1, 1, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.5,
            x_max: 1.2000000000000002,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (3, 1, 1, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [152, 159, 166, 152, 159, 166, 236, 243, 250, 236, 243, 250],
        },
        mask: Some([1, 1, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.2000000000000002,
            y_max: 1.6,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 1, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [26, 33, 40, 26, 33, 40, 152, 159, 166, 152, 159, 166],
        },
        mask: Some([0, 0, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.5,
            y_min: 0.3333333333333333,
            x_max: 2.0,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 0.6666666666666666,
            label: "corner",
          },
        ],
      }
    (1, 0, 1, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [47, 54, 61, 47, 54, 61, 173, 180, 187, 173, 180, 187],
        },
        mask: Some([0, 0, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.3333333333333333,
            x_max: 2.0,
            y_max: 1.7333333333333334,
            label: "main",
          },
        ],
      }
    (2, 0, 1, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [68, 75, 82, 68, 75, 82, 194, 201, 208, 194, 201, 208],
        },
        mask: Some([1, 1, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.3333333333333333,
            x_max: 2.0,
            y_max: 1.7333333333333334,
            label: "main",
          },
        ],
      }
    (3, 0, 1, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [89, 96, 103, 89, 96, 103, 215, 222, 229, 215, 222, 229],
        },
        mask: Some([1, 1, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.3333333333333333,
            x_max: 1.2000000000000002,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.3333333333333333,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [5, 12, 19, 26, 33, 40, 5, 12, 19, 26, 33, 40],
        },
        mask: Some([0, 0, 0, 0]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.25,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 2.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [89, 96, 103, 110, 117, 124, 89, 96, 103, 110, 117, 124],
        },
        mask: Some([2, 3, 2, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.25,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (0, 2, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [173, 180, 187, 194, 201, 208, 173, 180, 187, 194, 201, 208],
        },
        mask: Some([2, 2, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.25,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.2000000000000002,
            label: "main",
          },
        ],
      }
    (1, 0, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [26, 33, 40, 47, 54, 61, 26, 33, 40, 47, 54, 61],
        },
        mask: Some([0, 1, 0, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 1, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [110, 117, 124, 131, 138, 145, 110, 117, 124, 131, 138, 145],
        },
        mask: Some([3, 3, 3, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 2, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [194, 201, 208, 215, 222, 229, 194, 201, 208, 215, 222, 229],
        },
        mask: Some([2, 4, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.2000000000000002,
            label: "main",
          },
        ],
      }
    (2, 0, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [47, 54, 61, 68, 75, 82, 47, 54, 61, 68, 75, 82],
        },
        mask: Some([1, 1, 1, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 1.6,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (2, 1, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [131, 138, 145, 152, 159, 166, 131, 138, 145, 152, 159, 166],
        },
        mask: Some([3, 1, 3, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.6,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (2, 2, 2, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [215, 222, 229, 236, 243, 250, 215, 222, 229, 236, 243, 250],
        },
        mask: Some([4, 4, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.6,
            y_max: 1.2000000000000002,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 2, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [5, 12, 19, 26, 33, 40, 89, 96, 103, 110, 117, 124],
        },
        mask: Some([0, 0, 2, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.25,
            y_min: 0.5,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 2, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [89, 96, 103, 110, 117, 124, 173, 180, 187, 194, 201, 208],
        },
        mask: Some([2, 3, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.25,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.6,
            label: "main",
          },
        ],
      }
    (1, 0, 2, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [26, 33, 40, 47, 54, 61, 110, 117, 124, 131, 138, 145],
        },
        mask: Some([0, 1, 3, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.5,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 1, 2, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [110, 117, 124, 131, 138, 145, 194, 201, 208, 215, 222, 229],
        },
        mask: Some([3, 3, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.6,
            label: "main",
          },
        ],
      }
    (2, 0, 2, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [47, 54, 61, 68, 75, 82, 131, 138, 145, 152, 159, 166],
        },
        mask: Some([1, 1, 3, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.5,
            x_max: 1.6,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (2, 1, 2, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [131, 138, 145, 152, 159, 166, 215, 222, 229, 236, 243, 250],
        },
        mask: Some([3, 1, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.6,
            y_max: 1.6,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.0,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 2, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [26, 33, 40, 47, 54, 61, 152, 159, 166, 173, 180, 187],
        },
        mask: Some([0, 0, 2, 2]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.25,
            y_min: 0.3333333333333333,
            x_max: 2.0,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 0.6666666666666666,
            label: "corner",
          },
        ],
      }
    (1, 0, 2, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [47, 54, 61, 68, 75, 82, 173, 180, 187, 194, 201, 208],
        },
        mask: Some([0, 1, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.3333333333333333,
            x_max: 2.0,
            y_max: 1.7333333333333334,
            label: "main",
          },
        ],
      }
    (2, 0, 2, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [68, 75, 82, 89, 96, 103, 194, 201, 208, 215, 222, 229],
        },
        mask: Some([1, 1, 4, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.3333333333333333,
            x_max: 1.6,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.0,
            y_min: 1.3333333333333333,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 3, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [10, 17, 24, 42, 49, 56, 10, 17, 24, 42, 49, 56],
        },
        mask: Some([0, 1, 0, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.16666666666666666,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.6666666666666666,
            y_max: 2.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 3, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [94, 101, 108, 126, 133, 140, 94, 101, 108, 126, 133, 140],
        },
        mask: Some([2, 3, 2, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.16666666666666666,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (0, 2, 3, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [178, 185, 192, 210, 217, 224, 178, 185, 192, 210, 217, 224],
        },
        mask: Some([2, 4, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.16666666666666666,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.2000000000000002,
            label: "main",
          },
        ],
      }
    (1, 0, 3, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [31, 38, 45, 63, 70, 77, 31, 38, 45, 63, 70, 77],
        },
        mask: Some([0, 1, 0, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 1.0,
            x_max: 1.7333333333333334,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 1, 3, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [115, 122, 129, 147, 154, 161, 115, 122, 129, 147, 154, 161],
        },
        mask: Some([3, 1, 3, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.7333333333333334,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 2, 3, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [199, 206, 213, 231, 238, 245, 199, 206, 213, 231, 238, 245],
        },
        mask: Some([2, 4, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.7333333333333334,
            y_max: 1.2000000000000002,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.3333333333333333,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 3, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [10, 17, 24, 42, 49, 56, 94, 101, 108, 126, 133, 140],
        },
        mask: Some([0, 1, 2, 3]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.16666666666666666,
            y_min: 0.5,
            x_max: 2.0,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.6666666666666666,
            y_max: 1.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 3, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [94, 101, 108, 126, 133, 140, 178, 185, 192, 210, 217, 224],
        },
        mask: Some([2, 3, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.16666666666666666,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 1.6,
            label: "main",
          },
        ],
      }
    (1, 0, 3, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [31, 38, 45, 63, 70, 77, 115, 122, 129, 147, 154, 161],
        },
        mask: Some([0, 1, 3, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.5,
            x_max: 1.7333333333333334,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (1, 1, 3, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [115, 122, 129, 147, 154, 161, 199, 206, 213, 231, 238, 245],
        },
        mask: Some([3, 1, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.7333333333333334,
            y_max: 1.6,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.3333333333333333,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 3, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [31, 38, 45, 63, 70, 77, 157, 164, 171, 189, 196, 203],
        },
        mask: Some([0, 1, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.16666666666666666,
            y_min: 0.3333333333333333,
            x_max: 2.0,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.6666666666666666,
            y_max: 0.6666666666666666,
            label: "corner",
          },
        ],
      }
    (1, 0, 3, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [52, 59, 66, 84, 91, 98, 178, 185, 192, 210, 217, 224],
        },
        mask: Some([0, 1, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.3333333333333333,
            x_max: 1.7333333333333334,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.3333333333333333,
            y_min: 1.3333333333333333,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 4, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [16, 23, 30, 58, 65, 72, 16, 23, 30, 58, 65, 72],
        },
        mask: Some([0, 1, 0, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.125,
            y_min: 1.0,
            x_max: 1.8,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.5,
            y_max: 2.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 4, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [100, 107, 114, 142, 149, 156, 100, 107, 114, 142, 149, 156],
        },
        mask: Some([3, 1, 3, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.125,
            y_min: 0.0,
            x_max: 1.8,
            y_max: 2.0,
            label: "main",
          },
        ],
      }
    (0, 2, 4, 1) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [184, 191, 198, 226, 233, 240, 184, 191, 198, 226, 233, 240],
        },
        mask: Some([2, 4, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.125,
            y_min: 0.0,
            x_max: 1.8,
            y_max: 1.2000000000000002,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.5,
            y_min: 0.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 4, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [16, 23, 30, 58, 65, 72, 100, 107, 114, 142, 149, 156],
        },
        mask: Some([0, 1, 3, 1]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.125,
            y_min: 0.5,
            x_max: 1.8,
            y_max: 2.0,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.5,
            y_max: 1.0,
            label: "corner",
          },
        ],
      }
    (0, 1, 4, 2) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [100, 107, 114, 142, 149, 156, 184, 191, 198, 226, 233, 240],
        },
        mask: Some([3, 1, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.125,
            y_min: 0.0,
            x_max: 1.8,
            y_max: 1.6,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 1.5,
            y_min: 1.0,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    (0, 0, 4, 3) =>
      @moonaug.Sample::{
        image: @moonaug.Image::{
          width: 2,
          height: 2,
          channels: 3,
          data: [37, 44, 51, 79, 86, 93, 163, 170, 177, 205, 212, 219],
        },
        mask: Some([0, 1, 2, 4]),
        boxes: [
          @moonaug.BBox::{
            x_min: 0.125,
            y_min: 0.3333333333333333,
            x_max: 1.8,
            y_max: 1.7333333333333334,
            label: "main",
          },
          @moonaug.BBox::{
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.5,
            y_max: 0.6666666666666666,
            label: "corner",
          },
          @moonaug.BBox::{
            x_min: 1.5,
            y_min: 1.3333333333333333,
            x_max: 2.0,
            y_max: 2.0,
            label: "edge",
          },
        ],
      }
    _ => fail("resolved random crop lies outside the independent oracle table")
  }
}

///|
test "random resized crop uses independent exhaustive window oracle and exact replay" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let transform = @moonaug.RandomResizedCrop(@moonaug.ResizedCropOptions::{
    width: 2,
    height: 2,
    scale_min: 0.3,
    scale_max: 0.95,
    ratio_min: 0.5,
    ratio_max: 2.0,
  })
  for seed = 0; seed < 24; seed = seed + 1 {
    let result = @moonaug.augment(
      input,
      [{ transform, probability: 1.0, }],
      seed,
      "independent-rrc",
    )
    assert_eq(
      result,
      @moonaug.augment(
        input,
        [{ transform, probability: 1.0, }],
        seed,
        "independent-rrc",
      ),
    )
    guard result.replay.operations
      is [@moonaug.Crop(x, y, width, height), @moonaug.Resize(2, 2)] else {
      fail("random resized crop must resolve to Crop followed by Resize")
    }
    extended_sample_close(
      result.sample,
      extended_crop_expected(x, y, width, height),
      1,
    )
    assert_eq(@moonaug.replay(input, result.replay), result.sample)
  }
}

///|
test "random resized crop impossible aspect ratio uses documented center fallback" {
  let input = @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
        131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
        236, 243, 250,
      ],
    },
    mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.25,
        y_min: 0.5,
        x_max: 3.6,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  }
  let transform = @moonaug.RandomResizedCrop(@moonaug.ResizedCropOptions::{
    width: 2,
    height: 2,
    scale_min: 1.0,
    scale_max: 1.0,
    ratio_min: 0.2,
    ratio_max: 0.2,
  })
  let result = @moonaug.augment(
    input,
    [@moonaug.Step::{ transform, probability: 1.0, }],
    20261026,
    "fallback",
  )
  assert_eq(result.replay.operations, [
    @moonaug.Crop(1, 0, 1, 3),
    @moonaug.Resize(2, 2),
  ])
  extended_sample_close(result.sample, extended_crop_expected(1, 0, 1, 3), 1)
}

///|
fn extended_points_close(
  actual : Array[@moonaug.Keypoint],
  expected : Array[@moonaug.Keypoint],
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for i = 0; i < actual.length(); i = i + 1 {
    assert_eq(
      (actual[i].label, actual[i].visible),
      (expected[i].label, expected[i].visible),
    )
    let dx = actual[i].x - expected[i].x
    let dy = actual[i].y - expected[i].y
    assert_true(
      dx >= -1.0e-10 && dx <= 1.0e-10 && dy >= -1.0e-10 && dy <= 1.0e-10,
    )
  }
}

///|
test "independent keypoint geometry: crop_pad_letterbox_preserves_visibility_history" {
  let input = @moonaug.ExtendedSample::{
    sample: @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 4,
        height: 3,
        channels: 3,
        data: [
          5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
          131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
          236, 243, 250,
        ],
      },
      mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.25,
          y_min: 0.5,
          x_max: 3.6,
          y_max: 2.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 1.0,
          y_max: 1.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 3.0,
          y_min: 2.0,
          x_max: 4.0,
          y_max: 3.0,
          label: "edge",
        },
      ],
    },
    keypoints: [
      @moonaug.Keypoint::{ x: 0.0, y: 0.0, label: "origin", visible: true, },
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "edge", visible: true, },
      @moonaug.Keypoint::{
        x: 1.0,
        y: 2.0,
        label: "already-hidden",
        visible: false,
      },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
      @moonaug.Keypoint::{ x: 2.0, y: 1.0, label: "center", visible: true, },
      @moonaug.Keypoint::{
        x: 0.5,
        y: 2.5,
        label: "cropped-out",
        visible: true,
      },
    ],
  }
  let result = @moonaug.augment_with_keypoints(
    input,
    [
      @moonaug.Step::{ transform: @moonaug.Crop(1, 0, 3, 2), probability: 1.0, },
      @moonaug.Step::{
        transform: @moonaug.Pad(@moonaug.Padding::{
          left: 1,
          top: 2,
          right: 0,
          bottom: 1,
          fill: [7, 29, 131],
          mask_value: 1000,
        }),
        probability: 1.0,
      },
      @moonaug.Step::{
        transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
          width: 7,
          height: 7,
          fill: [23, 47, 91],
          mask_value: 1024,
          scale_up: true,
        }),
        probability: 1.0,
      },
    ],
    20261026,
    "points",
  )
  extended_sample_close(
    result.sample.sample,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 7,
        height: 7,
        channels: 3,
        data: [
          7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131,
          23, 47, 91, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131,
          7, 29, 131, 23, 47, 91, 7, 29, 131, 10, 30, 118, 13, 31, 106, 17, 35, 110,
          21, 39, 114, 24, 42, 117, 23, 47, 91, 7, 29, 131, 17, 31, 86, 30, 37, 44,
          43, 50, 57, 58, 65, 72, 68, 75, 82, 23, 47, 91, 7, 29, 131, 47, 61, 116,
          90, 97, 104, 104, 111, 118, 118, 125, 132, 128, 135, 142, 23, 47, 91, 7,
          29, 131, 36, 54, 129, 68, 81, 129, 76, 89, 137, 84, 97, 145, 90, 103, 151,
          23, 47, 91, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131, 7, 29, 131,
          7, 29, 131, 23, 47, 91,
        ],
      },
      mask: Some([
        1000, 1000, 1000, 1000, 1000, 1000, 1024, 1000, 1000, 1000, 1000, 1000, 1000,
        1024, 1000, 1000, 1000, 1000, 1000, 1000, 1024, 1000, 0, 0, 1, 1, 1, 1024,
        1000, 3, 3, 3, 1, 1, 1024, 1000, 3, 3, 3, 1, 1, 1024, 1000, 1000, 1000, 1000,
        1000, 1000, 1024,
      ]),
      boxes: [
        @moonaug.BBox::{
          x_min: 1.5,
          y_min: 3.5,
          x_max: 5.4,
          y_max: 5.6,
          label: "main",
        },
      ],
    },
    1,
  )
  extended_points_close(result.sample.keypoints, [
    @moonaug.Keypoint::{ x: 0.0, y: 2.8, label: "origin", visible: false, },
    @moonaug.Keypoint::{ x: 6.0, y: 7.0, label: "edge", visible: false, },
    @moonaug.Keypoint::{
      x: 1.5,
      y: 5.6,
      label: "already-hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 5.550000000000001,
      y: 3.08,
      label: "inside",
      visible: true,
    },
    @moonaug.Keypoint::{
      x: 3.0,
      y: 4.199999999999999,
      label: "center",
      visible: true,
    },
    @moonaug.Keypoint::{
      x: 0.75,
      y: 6.3,
      label: "cropped-out",
      visible: false,
    },
  ])
  assert_eq(result.replay.version, 2)
  let restored = @moonaug.replay_from_json(
    @moonaug.replay_to_json(result.replay),
  )
  assert_eq(@moonaug.replay_with_keypoints(input, restored), result.sample)
  assert_eq(input, @moonaug.ExtendedSample::{
    sample: @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 4,
        height: 3,
        channels: 3,
        data: [
          5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
          131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
          236, 243, 250,
        ],
      },
      mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.25,
          y_min: 0.5,
          x_max: 3.6,
          y_max: 2.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 1.0,
          y_max: 1.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 3.0,
          y_min: 2.0,
          x_max: 4.0,
          y_max: 3.0,
          label: "edge",
        },
      ],
    },
    keypoints: [
      @moonaug.Keypoint::{ x: 0.0, y: 0.0, label: "origin", visible: true, },
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "edge", visible: true, },
      @moonaug.Keypoint::{
        x: 1.0,
        y: 2.0,
        label: "already-hidden",
        visible: false,
      },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
      @moonaug.Keypoint::{ x: 2.0, y: 1.0, label: "center", visible: true, },
      @moonaug.Keypoint::{
        x: 0.5,
        y: 2.5,
        label: "cropped-out",
        visible: true,
      },
    ],
  })
}

///|
test "batch random geometry is keyed by ID, replay records align by ID" {
  let input = @moonaug.ExtendedSample::{
    sample: @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 4,
        height: 3,
        channels: 3,
        data: [
          5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124,
          131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229,
          236, 243, 250,
        ],
      },
      mask: Some([0, 0, 1, 1, 2, 3, 3, 1, 2, 2, 4, 4]),
      boxes: [
        @moonaug.BBox::{
          x_min: 0.25,
          y_min: 0.5,
          x_max: 3.6,
          y_max: 2.6,
          label: "main",
        },
        @moonaug.BBox::{
          x_min: 0.0,
          y_min: 0.0,
          x_max: 1.0,
          y_max: 1.0,
          label: "corner",
        },
        @moonaug.BBox::{
          x_min: 3.0,
          y_min: 2.0,
          x_max: 4.0,
          y_max: 3.0,
          label: "edge",
        },
      ],
    },
    keypoints: [
      @moonaug.Keypoint::{ x: 0.0, y: 0.0, label: "origin", visible: true, },
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "edge", visible: true, },
      @moonaug.Keypoint::{
        x: 1.0,
        y: 2.0,
        label: "already-hidden",
        visible: false,
      },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
      @moonaug.Keypoint::{ x: 2.0, y: 1.0, label: "center", visible: true, },
      @moonaug.Keypoint::{
        x: 0.5,
        y: 2.5,
        label: "cropped-out",
        visible: true,
      },
    ],
  }
  let items = [
    @moonaug.BatchItem::{ sample_id: "alpha", input, },
    @moonaug.BatchItem::{ sample_id: "beta", input, },
  ]
  let pipeline = [
    @moonaug.Step::{ transform: @moonaug.RandomCrop(3, 2), probability: 1.0, },
    @moonaug.Step::{ transform: @moonaug.HFlip, probability: 0.45, },
    @moonaug.Step::{
      transform: @moonaug.Letterbox(@moonaug.LetterboxOptions::{
        width: 5,
        height: 4,
        fill: [23, 47, 91],
        mask_value: 1024,
        scale_up: true,
      }),
      probability: 1.0,
    },
  ]
  let outputs = @moonaug.augment_batch(items, pipeline, 20261026)
  assert_eq(outputs[0].sample_id, "alpha")
  assert_eq(outputs[1].sample_id, "beta")
  assert_eq(
    outputs[0].output,
    @moonaug.augment_with_keypoints(input, pipeline, 20261026, "alpha"),
  )
  assert_eq(
    outputs[1].output,
    @moonaug.augment_with_keypoints(input, pipeline, 20261026, "beta"),
  )
  let reversed = @moonaug.augment_batch(
    [items[1], items[0]],
    pipeline,
    20261026,
  )
  assert_eq(reversed, [outputs[1], outputs[0]])
  let records = [
    @moonaug.BatchReplay::{
      sample_id: "alpha",
      replay: outputs[0].output.replay,
    },
    @moonaug.BatchReplay::{
      sample_id: "beta",
      replay: outputs[1].output.replay,
    },
  ]
  assert_eq(@moonaug.replay_batch([items[1], items[0]], records), reversed)
}
