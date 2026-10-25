// Generated from independent PyTorch rasters and NumPy edge/batch geometry.

///|
fn interop_reference_close(
  actual : Array[Double],
  expected : Array[Double],
) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for i = 0; i < actual.length(); i = i + 1 {
    let d = actual[i] - expected[i]
    assert_true(d >= -1.0e-9 && d <= 1.0e-9)
  }
}

///|
fn interop_reference_sample(
  actual : @moonaug.Sample,
  expected : @moonaug.Sample,
) -> Unit raise {
  assert_eq(
    (actual.image.width, actual.image.height, actual.image.channels),
    (expected.image.width, expected.image.height, expected.image.channels),
  )
  assert_eq(actual.image.data.length(), expected.image.data.length())
  for i = 0; i < actual.image.data.length(); i = i + 1 {
    let d = actual.image.data[i].to_int() - expected.image.data[i].to_int()
    assert_true(d >= -1 && d <= 1)
  }
  assert_eq(actual.mask, expected.mask)
  assert_eq(actual.boxes.length(), expected.boxes.length())
  for i = 0; i < actual.boxes.length(); i = i + 1 {
    let a = actual.boxes[i]
    let b = expected.boxes[i]
    assert_true(a.x_min < a.x_max && a.y_min < a.y_max)
    assert_eq(a.label, b.label)
    interop_reference_close([a.x_min, a.y_min, a.x_max, a.y_max], [
      b.x_min,
      b.y_min,
      b.x_max,
      b.y_max,
    ])
  }
}

///|
test "independent interop warp: identity_different_output_size_false" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  let result = @interop.warp_extended(
    input,
    theta,
    5,
    6,
    align_corners=false,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 6,
      height: 5,
      channels: 3,
      data: [
        3, 8, 13, 12, 18, 24, 24, 29, 35, 35, 40, 46, 46, 52, 57, 45, 50, 55, 32,
        38, 44, 49, 56, 63, 63, 70, 77, 77, 84, 91, 91, 98, 105, 85, 90, 96, 74,
        80, 86, 100, 106, 114, 114, 121, 128, 127, 134, 141, 142, 148, 156, 127,
        132, 138, 116, 122, 128, 150, 157, 164, 164, 171, 178, 178, 185, 192, 192,
        199, 206, 169, 174, 180, 115, 120, 125, 147, 152, 158, 158, 164, 169, 169,
        175, 180, 180, 186, 192, 157, 162, 167,
      ],
    },
    mask: Some([
      0, 0, 1, 4, 4, 2147483647, 0, 0, 1, 4, 4, 2147483647, 2, 2, 7, 4, 4, 1, 2,
      2, 2, 1024, 1024, 4, 2, 2, 2, 1024, 1024, 4,
    ]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.375,
        y_min: 0.8333333333333334,
        x_max: 5.4,
        y_max: 4.333333333333334,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.5,
        y_max: 1.6666666666666667,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 4.5,
        y_min: 3.3333333333333335,
        x_max: 6.0,
        y_max: 5.0,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{ x: 0.0, y: 0.0, label: "origin", visible: true, },
    @moonaug.Keypoint::{ x: 6.0, y: 5.0, label: "far-edge", visible: true, },
    @moonaug.Keypoint::{
      x: 1.5,
      y: 3.3333333333333335,
      label: "hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 5.550000000000001,
      y: 0.33333333333333337,
      label: "inside",
      visible: true,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      5,
      6,
      align_corners=false,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: reflection_false" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  let result = @interop.warp_extended(
    input,
    theta,
    3,
    4,
    align_corners=false,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        68, 75, 82, 47, 54, 61, 26, 33, 40, 5, 12, 19, 152, 159, 166, 131, 138, 145,
        110, 117, 124, 89, 96, 103, 236, 243, 250, 215, 222, 229, 194, 201, 208,
        173, 180, 187,
      ],
    },
    mask: Some([2147483647, 4, 1, 0, 1, 4, 7, 2, 4, 1024, 2, 2]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.3999999999999999,
        y_min: 0.5,
        x_max: 3.75,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 0.0,
        x_max: 4.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 2.0,
        x_max: 1.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{ x: 4.0, y: 0.0, label: "origin", visible: true, },
    @moonaug.Keypoint::{ x: 0.0, y: 3.0, label: "far-edge", visible: true, },
    @moonaug.Keypoint::{ x: 3.0, y: 2.0, label: "hidden", visible: false, },
    @moonaug.Keypoint::{
      x: 0.2999999999999998,
      y: 0.2,
      label: "inside",
      visible: true,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      3,
      4,
      align_corners=false,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: rotation_scale_translation_false" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [0.82, -0.23, 0.27, 0.17, 0.91, -0.19]
  let result = @interop.warp_extended(
    input,
    theta,
    4,
    5,
    align_corners=false,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 5,
      height: 4,
      channels: 3,
      data: [
        13, 17, 20, 24, 28, 33, 38, 43, 48, 52, 57, 62, 18, 20, 22, 37, 44, 51, 59,
        66, 73, 82, 89, 96, 104, 111, 118, 62, 66, 70, 89, 96, 103, 112, 119, 126,
        134, 141, 148, 156, 163, 170, 133, 138, 143, 142, 149, 156, 164, 171, 178,
        187, 194, 201, 209, 216, 223, 230, 237, 243,
      ],
    },
    mask: Some([
      987654, 4, 4, 2147483647, 987654, 1, 1, 4, 2147483647, 1, 7, 7, 4, 1, 1, 2,
      2, 1024, 1024, 4,
    ]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.7750753427564833,
        x_max: 4.711469077634874,
        y_max: 4.0,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 3.5494927628507154,
        y_min: 2.7768581009380697,
        x_max: 5.0,
        y_max: 4.0,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{
      x: -1.772252642302305,
      y: 0.8582707245638607,
      label: "origin",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 5.486119954157646,
      y: 4.169107347510505,
      label: "far-edge",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 0.652510717772401,
      y: 3.4262914385160657,
      label: "hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 3.684791374846131,
      y: 0.33575279086548615,
      label: "inside",
      visible: true,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      4,
      5,
      align_corners=false,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: partly_outside_mask_fill_false" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [0.9, 0.1, 1.2, -0.1, 1.0, 0.4]
  let result = @interop.warp_extended(
    input,
    theta,
    5,
    3,
    align_corners=false,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 3,
      height: 5,
      channels: 3,
      data: [
        100, 107, 114, 26, 28, 30, 0, 0, 0, 152, 159, 166, 27, 29, 30, 0, 0, 0, 205,
        212, 219, 20, 21, 22, 0, 0, 0, 162, 167, 172, 4, 4, 4, 0, 0, 0, 23, 24, 25,
        0, 0, 0, 0, 0, 0,
      ],
    },
    mask: Some([
      2147483647, 987654, 987654, 1, 987654, 987654, 4, 987654, 987654, 4, 987654,
      987654, 987654, 987654, 987654,
    ]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0164835164835169,
        y_max: 3.2142857142857135,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.24725274725274726,
        y_min: 2.1428571428571423,
        x_max: 1.1813186813186816,
        y_max: 3.928571428571428,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{
      x: -1.8956043956043955,
      y: -1.5659340659340657,
      label: "origin",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 1.0714285714285716,
      y: 3.928571428571428,
      label: "far-edge",
      visible: true,
    },
    @moonaug.Keypoint::{
      x: -1.291208791208791,
      y: 1.8681318681318677,
      label: "hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 1.1318681318681323,
      y: -0.7280219780219779,
      label: "inside",
      visible: false,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      5,
      3,
      align_corners=false,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: identity_different_output_size_true" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  let result = @interop.warp_extended(
    input,
    theta,
    5,
    6,
    align_corners=true,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 6,
      height: 5,
      channels: 3,
      data: [
        5, 12, 19, 18, 25, 32, 30, 37, 44, 43, 50, 57, 55, 62, 69, 68, 75, 82, 47,
        54, 61, 60, 67, 74, 72, 79, 86, 85, 92, 99, 97, 104, 111, 110, 117, 124,
        89, 96, 103, 102, 109, 116, 114, 121, 128, 127, 134, 141, 139, 146, 153,
        152, 159, 166, 131, 138, 145, 144, 151, 158, 156, 163, 170, 169, 176, 183,
        181, 188, 195, 194, 201, 208, 173, 180, 187, 186, 193, 200, 198, 205, 212,
        211, 218, 225, 223, 230, 237, 236, 243, 250,
      ],
    },
    mask: Some([
      0, 1, 1, 4, 4, 2147483647, 0, 1, 1, 4, 4, 2147483647, 2, 7, 7, 4, 4, 1, 2,
      2, 2, 1024, 1024, 4, 2, 2, 2, 1024, 1024, 4,
    ]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.08333333333333337,
        y_min: 0.5,
        x_max: 5.666666666666667,
        y_max: 4.7,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.3333333333333333,
        y_max: 1.5,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 4.666666666666667,
        y_min: 3.5,
        x_max: 6.0,
        y_max: 5.0,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{
      x: -0.33333333333333326,
      y: -0.5,
      label: "origin",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 6.333333333333333,
      y: 5.5,
      label: "far-edge",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 1.3333333333333333,
      y: 3.5,
      label: "hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 5.833333333333333,
      y: -0.09999999999999998,
      label: "inside",
      visible: false,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      5,
      6,
      align_corners=true,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: reflection_true" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
  let result = @interop.warp_extended(
    input,
    theta,
    3,
    4,
    align_corners=true,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 4,
      height: 3,
      channels: 3,
      data: [
        68, 75, 82, 47, 54, 61, 26, 33, 40, 5, 12, 19, 152, 159, 166, 131, 138, 145,
        110, 117, 124, 89, 96, 103, 236, 243, 250, 215, 222, 229, 194, 201, 208,
        173, 180, 187,
      ],
    },
    mask: Some([2147483647, 4, 1, 0, 1, 4, 7, 2, 4, 1024, 2, 2]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.3999999999999999,
        y_min: 0.5,
        x_max: 3.75,
        y_max: 2.6,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 3.0,
        y_min: 0.0,
        x_max: 4.0,
        y_max: 1.0,
        label: "corner",
      },
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 2.0,
        x_max: 1.0,
        y_max: 3.0,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{ x: 4.0, y: 0.0, label: "origin", visible: true, },
    @moonaug.Keypoint::{ x: 0.0, y: 3.0, label: "far-edge", visible: true, },
    @moonaug.Keypoint::{ x: 3.0, y: 2.0, label: "hidden", visible: false, },
    @moonaug.Keypoint::{
      x: 0.2999999999999998,
      y: 0.2,
      label: "inside",
      visible: true,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      3,
      4,
      align_corners=true,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: rotation_scale_translation_true" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [0.82, -0.23, 0.27, 0.17, 0.91, -0.19]
  let result = @interop.warp_extended(
    input,
    theta,
    4,
    5,
    align_corners=true,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 5,
      height: 4,
      channels: 3,
      data: [
        19, 24, 30, 32, 38, 43, 47, 53, 60, 64, 71, 78, 38, 42, 46, 50, 57, 64, 70,
        77, 84, 90, 97, 104, 110, 117, 124, 94, 99, 104, 96, 103, 110, 116, 123,
        130, 136, 143, 150, 156, 163, 170, 172, 179, 186, 142, 149, 156, 162, 169,
        176, 182, 189, 196, 202, 209, 216, 222, 229, 236,
      ],
    },
    mask: Some([
      1, 4, 4, 2147483647, 2147483647, 1, 1, 4, 1, 1, 7, 7, 4, 4, 1, 2, 2, 1024,
      1024, 4,
    ]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.47262192792563373,
        x_max: 5.0,
        y_max: 4.0,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 3.823485716711235,
        y_min: 2.735451419839552,
        x_max: 5.0,
        y_max: 4.0,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{
      x: -1.9832123604567256,
      y: 0.4688017318222337,
      label: "origin",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 5.954306209940999,
      y: 4.301731822233542,
      label: "far-edge",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 0.7333715352943673,
      y: 3.384884757417548,
      label: "hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 3.8506515556687466,
      y: -0.01890997071182976,
      label: "inside",
      visible: false,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      4,
      5,
      align_corners=true,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent interop warp: partly_outside_mask_fill_true" {
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  }
  let theta = [0.9, 0.1, 1.2, -0.1, 1.0, 0.4]
  let result = @interop.warp_extended(
    input,
    theta,
    5,
    3,
    align_corners=true,
    min_visibility=0.2,
    mask_fill=987654,
  )
  interop_reference_sample(result.sample, @moonaug.Sample::{
    image: @moonaug.Image::{
      width: 3,
      height: 5,
      channels: 3,
      data: [
        85, 92, 99, 86, 92, 98, 0, 0, 0, 128, 135, 142, 111, 117, 122, 0, 0, 0, 172,
        179, 186, 130, 135, 140, 0, 0, 0, 216, 223, 230, 142, 147, 151, 0, 0, 0,
        109, 112, 116, 78, 80, 83, 0, 0, 0,
      ],
    },
    mask: Some([
      4, 2147483647, 987654, 4, 1, 987654, 1024, 1, 987654, 1024, 4, 987654, 1024,
      4, 987654,
    ]),
    boxes: [
      @moonaug.BBox::{
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.5073260073260077,
        y_max: 3.855311355311356,
        label: "main",
      },
      @moonaug.BBox::{
        x_min: 0.7930402930402929,
        y_min: 2.580586080586081,
        x_max: 1.6355311355311355,
        y_max: 4.705128205128205,
        label: "edge",
      },
    ],
  })
  let expected_points = [
    @moonaug.Keypoint::{
      x: -1.0750915750915748,
      y: -1.8150183150183148,
      label: "origin",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 1.5256410256410255,
      y: 4.705128205128205,
      label: "far-edge",
      visible: true,
    },
    @moonaug.Keypoint::{
      x: -0.562271062271062,
      y: 2.2875457875457883,
      label: "hidden",
      visible: false,
    },
    @moonaug.Keypoint::{
      x: 1.613553113553114,
      y: -0.877289377289377,
      label: "inside",
      visible: false,
    },
  ]
  assert_eq(result.keypoints.length(), expected_points.length())
  for i = 0; i < result.keypoints.length(); i = i + 1 {
    let a = result.keypoints[i]
    let b = expected_points[i]
    assert_eq((a.label, a.visible), (b.label, b.visible))
    interop_reference_close([a.x, a.y], [b.x, b.y])
  }
  assert_eq(
    @interop.warp_sample(
      input.sample,
      theta,
      5,
      3,
      align_corners=true,
      min_visibility=0.2,
      mask_fill=987654,
    ),
    result.sample,
  )
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
      @moonaug.Keypoint::{ x: 4.0, y: 3.0, label: "far-edge", visible: true, },
      @moonaug.Keypoint::{ x: 1.0, y: 2.0, label: "hidden", visible: false, },
      @moonaug.Keypoint::{ x: 3.7, y: 0.2, label: "inside", visible: true, },
    ],
  })
}

///|
test "independent NumPy NCHW batch normalization and exact image round trip" {
  let images = [
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
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
    }.image,
    @moonaug.Sample::{
      image: @moonaug.Image::{
        width: 4,
        height: 3,
        channels: 3,
        data: [
          236, 243, 250, 215, 222, 229, 194, 201, 208, 173, 180, 187, 152, 159, 166,
          131, 138, 145, 110, 117, 124, 89, 96, 103, 68, 75, 82, 47, 54, 61, 26,
          33, 40, 5, 12, 19,
        ],
      },
      mask: Some([0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]),
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
    }.image,
  ]
  let packed = @interop.images_to_tensor(images)
  assert_eq(packed.shape(), [2, 3, 3, 4])
  interop_reference_close(packed.data(), [
    0.0196078431372549, 0.10196078431372549, 0.1843137254901961, 0.26666666666666666,
    0.34901960784313724, 0.43137254901960786, 0.5137254901960784, 0.596078431372549,
    0.6784313725490196, 0.7607843137254902, 0.8431372549019608, 0.9254901960784314,
    0.047058823529411764, 0.12941176470588237, 0.21176470588235294, 0.29411764705882354,
    0.3764705882352941, 0.4588235294117647, 0.5411764705882353, 0.6235294117647059,
    0.7058823529411765, 0.788235294117647, 0.8705882352941177, 0.9529411764705882,
    0.07450980392156863, 0.1568627450980392, 0.23921568627450981, 0.3215686274509804,
    0.403921568627451, 0.48627450980392156, 0.5686274509803921, 0.6509803921568628,
    0.7333333333333333, 0.8156862745098039, 0.8980392156862745, 0.9803921568627451,
    0.9254901960784314, 0.8431372549019608, 0.7607843137254902, 0.6784313725490196,
    0.596078431372549, 0.5137254901960784, 0.43137254901960786, 0.34901960784313724,
    0.26666666666666666, 0.1843137254901961, 0.10196078431372549, 0.0196078431372549,
    0.9529411764705882, 0.8705882352941177, 0.788235294117647, 0.7058823529411765,
    0.6235294117647059, 0.5411764705882353, 0.4588235294117647, 0.3764705882352941,
    0.29411764705882354, 0.21176470588235294, 0.12941176470588237, 0.047058823529411764,
    0.9803921568627451, 0.8980392156862745, 0.8156862745098039, 0.7333333333333333,
    0.6509803921568628, 0.5686274509803921, 0.48627450980392156, 0.403921568627451,
    0.3215686274509804, 0.23921568627450981, 0.1568627450980392, 0.07450980392156863,
  ])
  let mean = [0.2, 0.4, 0.6]
  let std = [0.3, 0.5, 0.7]
  let normalized = @interop.normalize(packed, mean, std)
  interop_reference_close(normalized.data(), [
    -0.6013071895424837, -0.3267973856209151, -0.05228758169934641, 0.22222222222222218,
    0.49673202614379075, 0.7712418300653595, 1.045751633986928, 1.3202614379084967,
    1.5947712418300655, 1.869281045751634, 2.1437908496732025, 2.4183006535947715,
    -0.7058823529411765, -0.5411764705882354, -0.37647058823529417, -0.21176470588235297,
    -0.04705882352941182, 0.11764705882352933, 0.2823529411764705, 0.44705882352941173,
    0.611764705882353, 0.776470588235294, 0.9411764705882353, 1.1058823529411763,
    -0.7507002801120449, -0.6330532212885154, -0.5154061624649859, -0.39775910364145656,
    -0.2801120448179271, -0.16246498599439774, -0.04481792717086835, 0.07282913165266112,
    0.19047619047619044, 0.3081232492997199, 0.4257703081232494, 0.5434173669467787,
    2.4183006535947715, 2.1437908496732025, 1.869281045751634, 1.5947712418300655,
    1.3202614379084967, 1.045751633986928, 0.7712418300653595, 0.49673202614379075,
    0.22222222222222218, -0.05228758169934641, -0.3267973856209151, -0.6013071895424837,
    1.1058823529411763, 0.9411764705882353, 0.776470588235294, 0.611764705882353,
    0.44705882352941173, 0.2823529411764705, 0.11764705882352933, -0.04705882352941182,
    -0.21176470588235297, -0.37647058823529417, -0.5411764705882354, -0.7058823529411765,
    0.5434173669467787, 0.4257703081232494, 0.3081232492997199, 0.19047619047619044,
    0.07282913165266112, -0.04481792717086835, -0.16246498599439774, -0.2801120448179271,
    -0.39775910364145656, -0.5154061624649859, -0.6330532212885154, -0.7507002801120449,
  ])
  interop_reference_close(
    @interop.denormalize(normalized, mean, std).data(),
    packed.data(),
  )
  assert_eq(@interop.tensor_to_images(packed), images)
  interop_reference_close(packed.data(), [
    0.0196078431372549, 0.10196078431372549, 0.1843137254901961, 0.26666666666666666,
    0.34901960784313724, 0.43137254901960786, 0.5137254901960784, 0.596078431372549,
    0.6784313725490196, 0.7607843137254902, 0.8431372549019608, 0.9254901960784314,
    0.047058823529411764, 0.12941176470588237, 0.21176470588235294, 0.29411764705882354,
    0.3764705882352941, 0.4588235294117647, 0.5411764705882353, 0.6235294117647059,
    0.7058823529411765, 0.788235294117647, 0.8705882352941177, 0.9529411764705882,
    0.07450980392156863, 0.1568627450980392, 0.23921568627450981, 0.3215686274509804,
    0.403921568627451, 0.48627450980392156, 0.5686274509803921, 0.6509803921568628,
    0.7333333333333333, 0.8156862745098039, 0.8980392156862745, 0.9803921568627451,
    0.9254901960784314, 0.8431372549019608, 0.7607843137254902, 0.6784313725490196,
    0.596078431372549, 0.5137254901960784, 0.43137254901960786, 0.34901960784313724,
    0.26666666666666666, 0.1843137254901961, 0.10196078431372549, 0.0196078431372549,
    0.9529411764705882, 0.8705882352941177, 0.788235294117647, 0.7058823529411765,
    0.6235294117647059, 0.5411764705882353, 0.4588235294117647, 0.3764705882352941,
    0.29411764705882354, 0.21176470588235294, 0.12941176470588237, 0.047058823529411764,
    0.9803921568627451, 0.8980392156862745, 0.8156862745098039, 0.7333333333333333,
    0.6509803921568628, 0.5686274509803921, 0.48627450980392156, 0.403921568627451,
    0.3215686274509804, 0.23921568627450981, 0.1568627450980392, 0.07450980392156863,
  ])
}

///|
test "categorical mask conversion preserves large IDs and batch offsets exactly" {
  let mask = [0, 1, 4, 2147483647, 2, 7, 4, 1, 2, 2, 1024, 4]
  let packed = @interop.mask_to_tensor(mask, 4, 3)
  assert_eq(packed.shape(), [1, 1, 3, 4])
  interop_reference_close(packed.data(), [
    0.0, 1.0, 4.0, 2147483647.0, 2.0, 7.0, 4.0, 1.0, 2.0, 2.0, 1024.0, 4.0,
  ])
  assert_eq(@interop.tensor_to_mask(packed), mask)
  let batch = @moonwarp.Tensor::new([2, 1, 3, 4], [
    4.0, 1024.0, 2.0, 2.0, 1.0, 4.0, 7.0, 2.0, 2147483647.0, 4.0, 1.0, 0.0, 0.0,
    1.0, 4.0, 2147483647.0, 2.0, 7.0, 4.0, 1.0, 2.0, 2.0, 1024.0, 4.0,
  ])
  assert_eq(@interop.tensor_to_mask(batch, batch_index=1), mask)
}
