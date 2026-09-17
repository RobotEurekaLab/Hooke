"""Bounded, provenance-preserving crops of selected PDS3 float32 terrain.

This adapter supports unrotated equirectangular, one-band PC_REAL products.
Unsupported labels and any missing pixels fail explicitly. Source spacing is
preserved; interpolated surfaces never become higher-resolution measurements.
"""

import argparse
import hashlib
import json
import re
import ssl
from pathlib import Path
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = {
    "lunar": dict(
        url="https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/HYGINUS/NAC_DTM_HYGINUS_E079N0063.IMG",
        product_id="NAC_DTM_HYGINUS_E079N0063",
        row=4086,
        column=1160,
        count=5,
        credit="NASA/GSFC/Arizona State University, LROC team",
        product_url="https://data.lroc.im-ldi.com/lroc/view_rdr_product/NAC_DTM_HYGINUS",
        precision_scope="5 m SOCET SET reported precision; see source README",
    ),
    "martian": dict(
        url="https://hirise-pds.lpl.arizona.edu/PDS/DTM/ESP/ORB_034300_034399/ESP_034394_1920_ESP_034249_1920/DTEEC_034394_1920_034249_1920_L01.IMG",
        product_id="DTEEC_034394_1920_034249_1920_L01",
        row=9734,
        column=3579,
        count=21,
        credit="NASA/JPL-Caltech/University of Arizona; DTM producer Joel Davis",
        product_url="https://hirise.lpl.arizona.edu/dtm/ESP_034394_1920",
        precision_scope="Source sampling is 1.0109257646541 m; no independent vertical accuracy qualification",
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def label_fields(data):
    text = data.decode("ascii", errors="replace")
    end = re.search(r"(?m)^END\s*$", text)
    if not end:
        raise ValueError("Complete attached PDS label is required")
    text = re.sub(r"/\*.*?\*/", "", text[: end.end()], flags=re.S)
    if not re.search(r"MAP_SCALE\s*=\s*[^\r\n]+<METERS/PIXEL>", text) or not re.search(
        r"A_AXIS_RADIUS\s*=\s*[^\r\n]+<KM>", text
    ):
        raise ValueError(
            "Selected PDS products require meters/pixel and kilometer radii"
        )
    fields = {}
    for key, value in re.findall(
        r"(?m)^\s*([A-Z_^][A-Z_0-9^]*)\s*=\s*([^\r\n]+)", text
    ):
        fields[key] = value.strip().split(" <")[0].strip('"')
    required = {
        "PDS_VERSION_ID": "PDS3",
        "SAMPLE_TYPE": "PC_REAL",
        "SAMPLE_BITS": "32",
        "BANDS": "1",
        "MAP_PROJECTION_TYPE": "EQUIRECTANGULAR",
        "MAP_PROJECTION_ROTATION": "0.0",
        "PROJECTION_LATITUDE_TYPE": "PLANETOCENTRIC",
    }
    for key, expected in required.items():
        if fields.get(key) != expected:
            raise ValueError(f"Unsupported PDS field {key}: {fields.get(key)}")
    if int(fields["RECORD_BYTES"]) != int(fields["LINE_SAMPLES"]) * 4:
        raise ValueError("Only contiguous float32 line records are supported")
    if fields.get("POSITIVE_LONGITUDE_DIRECTION") != "EAST":
        raise ValueError("East-positive longitude is required")
    return fields


def fetch_range(url, start, length, context):
    request = Request(url, headers={"Range": f"bytes={start}-{start+length-1}"})
    with urlopen(request, timeout=30, context=context) as response:
        payload = response.read(length + 1)
        content_range = response.headers.get("Content-Range", "")
        if (
            response.status != 206
            or not content_range.startswith(f"bytes {start}-{start+length-1}/")
            or len(payload) != length
        ):
            raise ValueError("Server did not return the exact requested PDS byte range")
        return payload, dict(
            final_url=response.url,
            content_range=content_range,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


def crop_product(world, output, context):
    product = PRODUCTS[world]
    output.mkdir(parents=True, exist_ok=True)
    prefix, header_response = fetch_range(product["url"], 0, 16384, context)
    fields = label_fields(prefix)
    if fields["PRODUCT_ID"] != product["product_id"]:
        raise ValueError("PDS product identity changed")
    record_bytes = int(fields["RECORD_BYTES"])
    image_offset = (int(fields["^IMAGE"]) - 1) * record_bytes
    header = (
        prefix[:image_offset]
        if image_offset <= len(prefix)
        else fetch_range(product["url"], 0, image_offset, context)[0]
    )
    (output / "source_label.lbl").write_bytes(header)
    count = product["count"]
    first_row, first_col = product["row"] - count // 2, product["column"] - count // 2
    if (
        first_row < 0
        or first_col < 0
        or first_row + count > int(fields["LINES"])
        or first_col + count > int(fields["LINE_SAMPLES"])
    ):
        raise ValueError("Crop lies outside the PDS raster")
    rows, requests = [], []
    for row in range(first_row, first_row + count):
        offset = image_offset + row * record_bytes + first_col * 4
        payload, response = fetch_range(product["url"], offset, count * 4, context)
        if (
            response["etag"] != header_response["etag"]
            or response["last_modified"] != header_response["last_modified"]
        ):
            raise ValueError("Source product changed during the crop download")
        values = np.frombuffer(payload, dtype="<f4")
        special = {
            int(value[3:-1], 16)
            for key, value in fields.items()
            if (key.startswith("CORE_") or key == "MISSING_CONSTANT")
            and value.startswith("16#")
        }
        if (
            not np.isfinite(values).all()
            or np.isin(values.view("<u4"), list(special)).any()
        ):
            raise ValueError("Crop contains PDS missing/saturated pixels")
        rows.append(values.astype(float))
        name = f"source_row_{row:06d}.bin"
        (output / name).write_bytes(payload)
        requests.append(
            dict(
                row_zero_based=row,
                start_byte=offset,
                length_bytes=len(payload),
                file=name,
                sha256=digest(payload),
                **response,
            )
        )
    scaling, offset = float(fields.get("SCALING_FACTOR", 1)), float(
        fields.get("OFFSET", 0)
    )
    absolute = np.asarray(rows) * scaling + offset
    origin_height = float(absolute[count // 2, count // 2])
    local = np.flipud(
        absolute - origin_height
    ).copy()  # Rows increase north in the local Z-up frame.
    spacing = float(fields["MAP_SCALE"])
    if not np.isfinite(local).all() or not spacing > 0:
        raise ValueError("Terrain units or elevations are invalid")
    np.save(output / "height_m.npy", local)
    radius_m = float(fields["A_AXIS_RADIUS"]) * 1000
    x = (product["column"] - float(fields["SAMPLE_PROJECTION_OFFSET"])) * spacing
    y = (float(fields["LINE_PROJECTION_OFFSET"]) - product["row"]) * spacing
    latitude = np.degrees(y / radius_m)
    longitude = (
        float(fields["CENTER_LONGITUDE"])
        + np.degrees(
            x / (radius_m * np.cos(np.radians(float(fields["CENTER_LATITUDE"]))))
        )
    ) % 360
    manifest = dict(
        schema_version=1,
        world=world,
        product=product,
        fields=fields,
        label_sha256=digest(header),
        downloaded_ranges=requests,
        height_sha256=digest((output / "height_m.npy").read_bytes()),
        shape=list(local.shape),
        source_spacing_m=spacing,
        extent_m=[(count - 1) * spacing] * 2,
        origin_latitude_deg=float(latitude),
        origin_longitude_deg=float(longitude),
        origin_source_elevation_m=origin_height,
        local_frame="east_north_up_center_pixel_origin",
        row_conversion="source_north_to_south_flipped_to_local_south_to_north",
        scale=scaling,
        offset=offset,
        interpolation="none_in_cached_pixels_piecewise_surface_in_physics",
        terms="public_domain_pds_scientific_archive_with_source_credit",
        terms_url="https://pds-ppi.igpp.ucla.edu/faq.jsp",
        no_full_product_hash_claim=True,
        qualification="download_and_crop_only_until_actual_physics_check",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "NOTICE.md").write_text(
        f"# PDS terrain crop\n\nSource: {product['product_url']}\n\nCredit: {product['credit']}. Product `{product['product_id']}`.\n\nPDS scientific archive data: {manifest['terms_url']}. Preserve source attribution; no agency endorsement is implied.\n\nHooke crops selected pixel centers, applies declared scale/offset, subtracts the center elevation and reverses the raster row direction. This is a derived local crop, not the complete source product. Source resolution and accuracy are unchanged.\n\n{product['precision_scope']}. No composition, dust, thermal properties or subpixel surface truth is inferred from this terrain.\n"
    )
    return manifest


def load_crop(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if digest((directory / "height_m.npy").read_bytes()) != metadata["height_sha256"]:
        raise ValueError("Terrain cache hash mismatch")
    height = np.load(directory / "height_m.npy", allow_pickle=False)
    if (
        list(height.shape) != metadata["shape"]
        or height.ndim != 2
        or min(height.shape) < 2
        or not np.isfinite(height).all()
    ):
        raise ValueError("Invalid terrain raster")
    return height, metadata


def add_heightfield(root, directory):
    height, metadata = load_crop(directory)
    floor = float(height.min())
    relief = max(float(np.ptp(height)), 1e-6)
    extent = metadata["extent_m"]
    ET.SubElement(
        root.find("asset"),
        "hfield",
        name="pds_terrain",
        nrow=str(height.shape[0]),
        ncol=str(height.shape[1]),
        size=f"{extent[0]/2} {extent[1]/2} {relief} .1",
        elevation=" ".join(map(str, ((height - floor) / relief).ravel())),
    )
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="pds_terrain",
        type="hfield",
        hfield="pds_terrain",
        pos=f"0 0 {floor}",
        rgba=".34 .33 .32 1" if metadata["world"] == "lunar" else ".42 .23 .14 1",
        friction=".8 .01 .001",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=SOURCE_ROOT / "assets/space/terrain"
    )
    parser.add_argument("--worlds", nargs="+", choices=PRODUCTS, default=list(PRODUCTS))
    parser.add_argument(
        "--ca-bundle", type=Path, default=Path("/etc/ssl/certs/ca-certificates.crt")
    )
    args = parser.parse_args()
    context = ssl.create_default_context(cafile=str(args.ca_bundle))
    for world in args.worlds:
        metadata = crop_product(world, args.output / world, context)
        print(
            json.dumps(
                dict(
                    world=world,
                    shape=metadata["shape"],
                    spacing_m=metadata["source_spacing_m"],
                    height_range_m=float(np.ptp(load_crop(args.output / world)[0])),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
