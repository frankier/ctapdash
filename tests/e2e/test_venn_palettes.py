"""Step-count controls, palette icons, themes, and the third data stream."""

from playwright.sync_api import expect
from tests.e2e.test_domains import MODELS, open_viewer, ready, state, choose


def test_step_counts_palette_menu_and_hull(page, dashboard_url):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    open_viewer(page, dashboard_url)
    expect(page.get_by_text("Step A", exact=True)).to_be_visible()
    expect(page.get_by_text("Step B", exact=True)).to_be_visible()
    expect(page.get_by_text("Step C", exact=True)).not_to_be_visible()
    expect(page.get_by_label("Show min-max hull", exact=True)).to_be_checked()
    previous = state(page)["version"]
    page.get_by_role("button", name="+", exact=True).click()
    ready(page, previous)
    expect(page.get_by_text("Step C", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="+", exact=True)).to_be_disabled()
    page.evaluate(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        m.lines_visible = true;
    }}""")
    page.wait_for_function(f"""() => {{
        const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
        return m.ready && m.step_count === 3 && m.line_source_c.data.ys.some(row => row.some(Number.isFinite));
    }}""")
    choose(page, "domain-mode", "intersection")
    assert state(page)["count"] == 225
    for palette, background, hull in [
        ("cym_paint", "#ffffff", "#dddddd"),
        ("cym_sub", "#ffffff", "#dddddd"),
        ("rbg_add", "#000000", "#222222"),
    ]:
        page.get_by_role("button", name="Palette:").click()
        options = page.get_by_role("menuitemradio")
        expect(options).to_have_count(3)
        for option in options.all():
            expect(option.locator("img")).to_have_js_property("complete", True)
            assert option.locator("img").evaluate("img => img.naturalWidth") > 0
        page.get_by_role("menuitemradio", name=palette, exact=True).click()
        page.wait_for_function(
            f"""([background, hull]) => {{
            const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
            const p = {MODELS}.find(m => m.name === 'comparison-plot' && Bokeh.index.find_one(m) != null);
            return m?.ready && m.palette[0] === background && m.hull_color === hull && p.background_fill_color === background;
        }}""",
            arg=[background, hull],
        )
    page.get_by_label("Show min-max hull", exact=True).uncheck()
    page.wait_for_function(
        f"() => !{MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null).hull_visible"
    )
    for _ in range(2):
        previous = state(page)["version"]
        page.get_by_role("button", name="−", exact=True).click()
        ready(page, previous)
    expect(page.get_by_role("button", name="−", exact=True)).to_be_disabled()
    expect(page.get_by_text("Step B", exact=True)).not_to_be_visible()
    assert state(page)["count"] == 500
    assert errors == []


def test_shader_palette_bits_and_hull_pixels(page, dashboard_url):
    from venn_ts.palettes import PALETTES, hull_color

    open_viewer(page, dashboard_url)
    # Run the actual linked production shader in an isolated WebGL canvas.
    # Eight columns exercise every bitset; separate rows test traces and gaps.
    for palette in PALETTES.values():
        for count in (1, 2, 3):
            for show_hull in (False, True):
                pixels = page.evaluate(
                    f"""([palette, hull, count, show_hull]) => {{
                    const m = {MODELS}.find(m => m.type === 'venn_ts.renderer.VennTimeSeriesRenderer' && Bokeh.index.find_one(m) != null);
                    const original = Bokeh.index.find_one(m).get_resources();
                    const canvas = document.createElement('canvas');
                    canvas.width = 8; canvas.height = 4;
                    const gl = canvas.getContext('webgl');
                    gl.getExtension('OES_texture_float');
                    const program = gl.createProgram();
                    for (const shader of original.gl.getAttachedShaders(original.program)) {{
                        const copy = gl.createShader(original.gl.getShaderParameter(shader, original.gl.SHADER_TYPE));
                        gl.shaderSource(copy, original.gl.getShaderSource(shader));
                        gl.compileShader(copy);
                        if (!gl.getShaderParameter(copy, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(copy));
                        gl.attachShader(program, copy);
                    }}
                    gl.linkProgram(program);
                    gl.useProgram(program);
                    const buffer = gl.createBuffer();
                    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
                    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
                    const position = gl.getAttribLocation(program, 'a_position');
                    gl.enableVertexAttribArray(position);
                    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
                    const rgba = color => [1,3,5].map(i=>parseInt(color.slice(i,i+2),16)/255).concat(1);
                    const f = (name, v) => gl.uniform1f(gl.getUniformLocation(program,name),v);
                    const i = (name, v) => gl.uniform1i(gl.getUniformLocation(program,name),v);
                    const v2 = (name, a, b) => gl.uniform2f(gl.getUniformLocation(program,name),a,b);
                    const ranges = new Float32Array(32), third = new Float32Array(32), valid = new Uint8Array(16);
                    for (let mask=0; mask<8; mask++) {{
                        for (let side=0; side<3; side++) {{
                            const low = (mask & (1<<side)) ? 1 : 21;
                            if (side<2) {{
                                ranges[mask*4+side*2] = low;
                                ranges[mask*4+side*2+1] = low+8;
                                valid[mask*2+side] = side<count ? 1 : 0;
                            }} else {{
                                third.set([low,low+8,count===3 ? 1 : 0,0], mask*4);
                            }}
                        }}
                    }}
                    const texture = (unit, name, data, format, type) => {{
                        gl.activeTexture(gl.TEXTURE0+unit);
                        gl.bindTexture(gl.TEXTURE_2D,gl.createTexture());
                        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
                        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
                        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
                        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
                        gl.texImage2D(gl.TEXTURE_2D,0,format,8,1,0,format,type,data);
                        i(name,unit);
                    }};
                    texture(0,'u_ranges',ranges,gl.RGBA,gl.FLOAT);
                    texture(1,'u_valid',valid,gl.LUMINANCE_ALPHA,gl.UNSIGNED_BYTE);
                    texture(2,'u_edges',ranges,gl.RGBA,gl.FLOAT);
                    texture(3,'u_ranges_c',third,gl.RGBA,gl.FLOAT);
                    v2('u_texture_size',8,1); v2('u_lookup_size',8,1);
                    v2('u_frame_origin',0,0); v2('u_frame_size',8,4);
                    f('u_entry_count',8); f('u_channel_row',0);
                    f('u_source_time_start',0); f('u_source_time_step',1);
                    f('u_x_start',0); f('u_x_end',8); f('u_y_start',0); f('u_y_end',40);
                    f('u_amplitude_scale',1); f('u_amplitude_offset',0);
                    i('u_irregular',0); i('u_hull_visible',show_hull ? 1 : 0);
                    gl.uniform4fv(gl.getUniformLocation(program,'u_palette[0]'),palette.flatMap(rgba));
                    gl.uniform4fv(gl.getUniformLocation(program,'u_hull_color'),rgba(hull));
                    gl.clearColor(...rgba(palette[0])); gl.clear(gl.COLOR_BUFFER_BIT);
                    gl.viewport(0,0,8,4); gl.drawArrays(gl.TRIANGLES,0,6);
                    const pixels = new Uint8Array(8*4*4);
                    gl.readPixels(0,0,8,4,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
                    const error=gl.getError();
                    gl.getExtension('WEBGL_lose_context')?.loseContext();
                    if (error) throw new Error(`WebGL error ${{error}}`);
                    return Array.from({{length:32}}, (_,n)=>'#'+Array.from(pixels.slice(n*4,n*4+3),v=>v.toString(16).padStart(2,'0')).join(''));
                }}""",
                    [palette, hull_color(palette[0]), count, show_hull],
                )
                mask = (1 << count) - 1
                assert pixels[:8] == [palette[i & mask] for i in range(8)]
                assert pixels[8:16] == [
                    hull_color(palette[0])
                    if show_hull and 0 < (i & mask) < mask
                    else palette[0]
                    for i in range(8)
                ]
                assert pixels[24:] == [palette[0]] * 8
