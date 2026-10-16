import pytest
from PIL import Image

from vision_rl.aircraft_data import (
    aircraft_pixel_sha256,
    load_aircraft_rgb,
    remove_banner,
    rgb_pixel_sha256,
)


def test_lazy_crop_removes_only_bottom20_and_hashes_exact_pixels(tmp_path):
    source = Image.new("RGB", (11, 30), (20, 30, 40))
    for y in range(10, 30):
        for x in range(11):
            source.putpixel((x, y), (255, 0, 0))
    path = tmp_path / "fixture.png"
    source.save(path)
    original_bytes = path.read_bytes()
    with load_aircraft_rgb(path) as result:
        assert result.size == (11, 10)
        assert result.tobytes() == bytes((20, 30, 40)) * 110
        assert aircraft_pixel_sha256(path) == rgb_pixel_sha256(result)
    assert path.read_bytes() == original_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fixture.png"]


def test_banner_removal_can_reveal_duplicates_hidden_by_different_banners(tmp_path):
    hashes, full = [], []
    for value in (10, 200):
        image = Image.new("RGB", (5, 25), (1, 2, 3))
        for y in range(5, 25):
            for x in range(5):
                image.putpixel((x, y), (value, value, value))
        path = tmp_path / f"{value}.png"
        image.save(path)
        full.append(rgb_pixel_sha256(image))
        hashes.append(aircraft_pixel_sha256(path))
    assert full[0] != full[1]
    assert hashes[0] == hashes[1]


def test_crop_rejects_small_image_or_unknown_orientation(tmp_path):
    with pytest.raises(ValueError, match="more than 20"):
        remove_banner(Image.new("RGB", (3, 20)))
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (30, 30))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match="orientation"):
        load_aircraft_rgb(path)
