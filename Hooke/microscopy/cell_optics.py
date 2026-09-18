"""Separate estimated phase-contrast and fluorescent cell observations."""

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from microscopy.cell_scene import CELL_CENTER, CELL_RADII, NEEDLE_BACK, NUCLEAR_RADII
from microscopy.optics import Microscope, OpticalCalibration
from microscopy.optical_cache import PlaneBlurCache
from microscopy.pipettes import HOLDING_PROFILE, INJECTION_PROFILE
from microscopy.cell_appearance import background_geometry, cell_fields
from microscopy.phase_contrast import PhaseContrastOptics


class CellMicroscope(Microscope):
    scale_bar_m = 20e-6

    def __init__(self):
        super().__init__(OpticalCalibration(field_width_m=160e-6, pixels=768, defocus_scale_m=.45e-6))
        self._plane_blur = PlaneBlurCache()
        self.phase_optics = PhaseContrastOptics()
        self._projection = lru_cache(maxsize=16)(self._projection)

    def public_calibration(self):
        return dict(super().public_calibration(),
                    imaging_model="scalar_thin_object_annular_phase_contrast",
                    observation_channel="synthetic_nuclear_fluorescence",
                    display_channels=["phase_contrast", "fluorescence"],
                    wavelength_m=self.phase_optics.wavelength_m,
                    numerical_aperture=self.phase_optics.numerical_aperture,
                    detector_read_noise_dn=.25,
                    empirical_defocus_scale_m=self.calibration.defocus_scale_m,
                    axial_response="Independent Gaussian object planes; not calibrated optical sectioning",
                    biological_texture_calibrated=False)

    def _projection(self, index, dx0, dy0, width, radii, nuclear, indentation_m):
        yy, xx = np.mgrid[:width, :width]
        dx = dx0+xx*self.calibration.pixel_size_m
        dy = dy0-yy*self.calibration.pixel_size_m
        angle = 0. if index == 0 else background_geometry(index)[1]
        u = np.cos(angle)*dx+np.sin(angle)*dy
        v = -np.sin(angle)*dx+np.cos(angle)*dy
        opd, absorption, interior, nucleus = cell_fields(
            u, v, radii, nuclear, index=index, indentation_m=indentation_m)
        contrast = self.phase_optics.intensity(opd, absorption, self.calibration.pixel_size_m)-1.
        edge = np.minimum.reduce([xx, yy, width-1-xx, width-1-yy])
        window = np.clip(edge/24., 0., 1.)
        contrast *= window*window*(3-2*window)
        for values in (contrast, interior, nucleus):
            values.setflags(write=False)
        return contrast, interior, nucleus

    def render(self, state, *, annotate=True, channel="phase_contrast"):
        if channel not in ("phase_contrast", "fluorescence"):
            raise ValueError("Unknown cell imaging channel")
        c = self.calibration
        yy, xx = np.mgrid[:c.pixels, :c.pixels]
        x = (xx-c.pixels/2)*c.pixel_size_m
        y = (c.pixels/2-yy)*c.pixel_size_m
        base = 196-9*((x/(80e-6))**2+(y/(80e-6))**2)
        if channel == "fluorescence":
            base = np.full_like(base, 9.)
        rgb = np.repeat(base[..., None], 3, axis=2)
        reference_height = state.get("optical_reference_height_m", CELL_CENTER[2])
        focal_height = reference_height+state["focus_m"]
        radii = np.asarray(state.get("cell_radii_m", CELL_RADII))
        nuclear = np.asarray(state.get("nuclear_radii_m", NUCLEAR_RADII))
        for i, centre in enumerate(state["centres_m"]):
            if channel == "fluorescence" and i != 0:
                continue  # Only the target is labelled in this experiment.
            cell_radii = radii if i == 0 else background_geometry(i)[0]
            sigma = abs(centre[2]-focal_height)/c.defocus_scale_m
            # Render a padded local patch so defocus is applied independently
            # to each object plane, including content outside the field edge.
            cx, cy = c.pixel(centre[:2])
            sharp_halfwidth = int(np.ceil(1.3*max(cell_radii[:2])/c.pixel_size_m+24))
            halfwidth = sharp_halfwidth+int(np.ceil(4*sigma))
            if sigma > c.pixels/4:
                continue
            left, top = int(np.floor(cx))-halfwidth, int(np.floor(cy))-halfwidth
            height = width = 2*halfwidth+1
            nr = nuclear if i == 0 else cell_radii*np.array([.33, .3, .5])
            offset = halfwidth-sharp_halfwidth
            region = np.s_[offset:offset+2*sharp_halfwidth+1,
                           offset:offset+2*sharp_halfwidth+1]
            contrast, interior, nucleus = self._projection(
                i, float((left+offset-c.pixels/2)*c.pixel_size_m-centre[0]),
                float((c.pixels/2-top-offset)*c.pixel_size_m-centre[1]),
                2*sharp_halfwidth+1, tuple(cell_radii), tuple(nr),
                state["indentation_m"] if i == 0 else 0.)
            if channel == "phase_contrast":
                # Exposure maps optical intensity to camera values. The dose
                # does not recolour the unstained transmission channel.
                signal = np.zeros((height, width))
                signal[region] = 48*contrast
            else:
                signal = np.zeros((height, width, 3))
                signal[region] = nucleus[..., None]*np.array([22., 45., 230.])
                tracer = min(1., state["delivered_pl"]/.5)
                signal[region] += tracer*interior[..., None]*np.array([5., 140., 18.])
            if sigma:
                widths = (sigma, sigma) if signal.ndim == 2 else (sigma, sigma, 0)
                signal = self._plane_blur(signal, sigma=widths, mode="constant", cval=0.)
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(c.pixels, left+width), min(c.pixels, top+height)
            if x1 > x0 and y1 > y0:
                visible = signal[y0-top:y1-top, x0-left:x1-left]
                rgb[y0:y1, x0:x1] += visible[..., None] if signal.ndim == 2 else visible
        rgb += np.random.default_rng(123).normal(0, .25, (c.pixels, c.pixels, 1))
        image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
        if channel == "phase_contrast":
            image = self.handling_tools(image, state, focal_height)
        if "holder_tip_m" in state and channel == "phase_contrast":
            layer = Image.new("RGBA", image.size)
            draw = ImageDraw.Draw(layer)
            holding_tip = np.asarray(state["holder_tip_m"][:2])
            self.draw_pipette(draw, holding_tip, np.asarray(state["holder_back"]), HOLDING_PROFILE)
            image = self.composite_tool(image, layer, abs(state["holder_tip_m"][2]-focal_height)/c.defocus_scale_m)
            draw = ImageDraw.Draw(image)
            if state["holding_sealed"] and annotate:
                px, py = c.pixel(holding_tip)
                draw.line((px, py-18, px, py+18), fill=(44, 154, 148), width=2)
        tip = np.asarray(state["tip_m"][:2])
        layer = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(layer)
        if channel == "phase_contrast":
            self.draw_pipette(draw, tip, NEEDLE_BACK, INJECTION_PROFILE)
        image = self.composite_tool(image, layer, abs(state["tip_m"][2]-focal_height)/c.defocus_scale_m)
        draw = ImageDraw.Draw(image)
        px, py = c.pixel(tip)
        if annotate and state["punctured"] and not state["withdrawn"]:
            draw.ellipse((px-5, py-5, px+5, py+5), outline=(225, 146, 74), width=2)
        if annotate:
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, c.pixels, 32), fill=(23, 33, 41))
            label = "Estimated phase contrast" if channel == "phase_contrast" else "Synthetic fluorescence | nuclear label + tracer"
            draw.text((14, 10), "CELL PHANTOM  |  "+label, fill="white")
            length = self.scale_bar_m/c.pixel_size_m
            ink = (30, 43, 48) if channel == "phase_contrast" else (225, 230, 235)
            draw.rectangle((25, c.pixels-32, 25+length, c.pixels-27), fill=ink)
            draw.text((25, c.pixels-50), "20 um", fill=ink)
            status = "WITHDRAWN" if state["withdrawn"] else "PUNCTURED" if state["punctured"] else "INTACT"
            draw.text((c.pixels-300, c.pixels-24),
                      f"{status} | {state['delivered_pl']:.3f} pL | t={state['time_s']:.2f}s", fill=ink)
        return image

    def handling_tools(self, image, state, focal_height):
        """Project fine probe and forceps geometry at their observed heights."""
        c = self.calibration
        if "probe_tip_m" in state:
            tip = np.asarray(state["probe_tip_m"])
            layer = Image.new("RGBA", image.size)
            draw = ImageDraw.Draw(layer)
            radius = state["probe_radius_m"]
            points = [tip[:2]+[x, y] for x, y in
                      ((0, radius), (-160e-6, 5e-6), (-160e-6, -5e-6), (0, -radius))]
            draw.polygon([c.pixel(p) for p in points], fill=(92, 92, 92, 220))
            px, py = c.pixel(tip[:2])
            r = radius/c.pixel_size_m
            draw.ellipse((px-r, py-r, px+r, py+r), fill=(90, 90, 90, 220))
            image = self.composite_tool(image, layer, abs(tip[2]-focal_height)/c.defocus_scale_m)
        for index, values in enumerate(state.get("jaw_centres_m", [])):
            centre = np.asarray(values)
            halfsize = np.asarray(state["jaw_halfsize_m"])
            layer = Image.new("RGBA", image.size)
            draw = ImageDraw.Draw(layer)
            points = [centre[:2]+[x*halfsize[0], y*halfsize[1]]
                      for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            draw.polygon([c.pixel(p) for p in points], fill=(95, 95, 95, 220))
            sign = 1 if index == 0 else -1
            draw.line((*c.pixel(centre[:2]+[halfsize[0], 0]),
                       *c.pixel(centre[:2]+[160e-6, sign*25e-6])),
                      fill=(108, 108, 108, 210), width=max(1, round(3e-6/c.pixel_size_m)))
            image = self.composite_tool(image, layer, abs(centre[2]-focal_height)/c.defocus_scale_m)
        return image

    def draw_pipette(self, draw, tip, back, profile):
        """Top-view silhouette from the same SI meridian used by the 3D mesh."""
        c = self.calibration
        direction = back[:2]/np.linalg.norm(back[:2])
        perpendicular = np.array([-direction[1], direction[0]])
        distances = np.r_[0., np.asarray(profile.sections)[1:, 0]]
        centres = tip+distances[:, None]*back[:2]
        for inner, fill, outline in ((False, (179, 179, 179, 64), (82, 82, 82, 200)),
                                     (True, (202, 202, 202, 20), (149, 149, 149, 128))):
            radii = profile.radius(distances, inner=inner)[:, None]*perpendicular
            polygon = [c.pixel(point) for point in np.r_[centres+radii, (centres-radii)[::-1]]]
            draw.polygon(polygon, fill=fill)
            draw.line(polygon+[polygon[0]], fill=outline, width=1)

    def composite_tool(self, image, layer, sigma):
        """Approximate a capillary as a single optical plane at its tip height."""
        sigma = float(sigma)
        if sigma > self.calibration.pixels/4:
            return image
        pixels = np.asarray(layer, dtype=float)
        alpha = pixels[..., 3:]/255.
        # Blur premultiplied colour and opacity separately to avoid dark
        # fringes at the glass boundary. PIL's box approximation stays fast
        # for a far-defocused tip; this is not a wave-optics PSF.
        colour = Image.fromarray((pixels[..., :3]*alpha).astype(np.uint8))
        opacity = layer.getchannel("A")
        if sigma:
            colour = colour.filter(ImageFilter.GaussianBlur(sigma))
            opacity = opacity.filter(ImageFilter.GaussianBlur(sigma))
        weight = np.asarray(opacity, dtype=float)[..., None]/255.
        result = np.asarray(image, dtype=float)*(1-weight)+np.asarray(colour, dtype=float)
        return Image.fromarray(result.clip(0,255).astype(np.uint8))

    def locate(self, image, sample):
        if sample != "cell":
            raise ValueError("Only the marked target cell is supported")
        rgb = np.asarray(image, dtype=float)
        red, green, blue = rgb.transpose(2, 0, 1)
        mask = (blue > 140) & (blue > red*1.35) & (blue > green*1.15)
        yy, xx = np.nonzero(mask)
        if len(xx) < 40:
            raise RuntimeError("Target nuclear marker is not visible; check focus and field")
        return self.calibration.position([xx.mean(), yy.mean()])
