def has_prisma_data(prisma: dict) -> bool:
    ident = prisma.get("identification", {})
    return any(
        (
            ident.get("ieee", 0),
            ident.get("crossref", 0),
            ident.get("semanticscholar", 0),
            prisma.get("after_dedup", 0),
            prisma.get("screened", 0),
            prisma.get("included", 0),
        )
    )


def build_prisma_svg(prisma: dict) -> str:
    """Build a PRISMA 2020 flow diagram as an SVG string from the prisma counts dict."""
    ident = prisma.get("identification", {})
    n_ieee = ident.get("ieee", 0)
    n_crossref = ident.get("crossref", 0)
    n_s2 = ident.get("semanticscholar", 0)
    n_total_id = n_ieee + n_crossref + n_s2
    n_dedup = prisma.get("after_dedup", 0)
    n_removed = n_total_id - n_dedup
    n_screened = prisma.get("screened", n_dedup)
    n_excl_screen = prisma.get("excluded_screening", 0)
    n_sought = prisma.get("sought_retrieval", max(0, n_screened - n_excl_screen))
    n_not_retrieved = prisma.get("not_retrieved", 0)
    n_assessed = prisma.get("assessed_eligibility", max(0, n_sought - n_not_retrieved))
    n_excl_elig = prisma.get("excluded_eligibility", 0)
    n_included = prisma.get("included", max(0, n_assessed - n_excl_elig))

    width = 760
    box_w, box_h = 280, 60
    excl_w = 220
    left_x = 80
    right_x = 430
    label_x = 20
    label_w = 50

    def box(x, y, w, h, text, excluded=False, sub=""):
        fill = "#fff5f5" if excluded else "#ffffff"
        stroke = "#c53030" if excluded else "#2c5282"
        lines = []
        for i, t in enumerate([text] + ([sub] if sub else [])):
            lines.append(
                f'<text x="{x+10}" y="{y+22+i*16}" font-size="12" '
                f'font-family="Arial" fill="#1a202c">{t}</text>'
            )
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            + "".join(lines)
        )

    def arrow(x1, y1, x2, y2):
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>'
        )

    def section_label(y, h, label):
        return (
            f'<rect x="{label_x}" y="{y}" width="{label_w}" height="{h}" rx="4" fill="#4a90d9"/>'
            f'<text x="{label_x+25}" y="{y+h//2}" font-size="11" font-family="Arial" fill="white" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'transform="rotate(-90, {label_x+25}, {y+h//2})">{label}</text>'
        )

    id_y = 45
    screening_y = id_y + box_h + 30
    sought_y = screening_y + box_h + 30
    eligibility_y = sought_y + box_h + 30
    included_y = eligibility_y + box_h + 30
    height = included_y + 85

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#fff;font-family:Arial,sans-serif;">'
        '<defs><marker id="arr" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#4a5568"/></marker></defs>'
        f'<text x="{width//2}" y="25" text-anchor="middle" font-size="15" '
        f'font-weight="bold" fill="#1a202c">PRISMA 2020 Flow Diagram</text>'
    )

    svg += section_label(id_y, 100, "Identification")
    svg += box(
        left_x,
        id_y,
        box_w,
        box_h,
        f"Records identified (n = {n_total_id})",
        sub=f"IEEE: {n_ieee}   Crossref: {n_crossref}",
    )
    svg += box(
        right_x,
        id_y,
        excl_w,
        box_h,
        f"Duplicates removed (n = {n_removed})",
        excluded=True,
    )
    svg += arrow(left_x + box_w, id_y + box_h // 2, right_x, id_y + box_h // 2)
    svg += arrow(left_x + box_w // 2, id_y + box_h, left_x + box_w // 2, id_y + box_h + 20)

    svg += section_label(screening_y, 100, "Screening")
    svg += box(left_x, screening_y, box_w, box_h, f"Records screened (n = {n_screened})")
    svg += box(
        right_x,
        screening_y,
        excl_w,
        box_h,
        f"Records excluded (n = {n_excl_screen})",
        excluded=True,
    )
    svg += arrow(left_x + box_w, screening_y + box_h // 2, right_x, screening_y + box_h // 2)
    svg += arrow(
        left_x + box_w // 2,
        screening_y + box_h,
        left_x + box_w // 2,
        screening_y + box_h + 20,
    )

    svg += box(
        left_x,
        sought_y,
        box_w,
        box_h,
        f"Reports sought for retrieval (n = {n_sought})",
    )
    svg += box(
        right_x,
        sought_y,
        excl_w,
        box_h,
        f"Not retrieved (n = {n_not_retrieved})",
        excluded=True,
    )
    svg += arrow(left_x + box_w, sought_y + box_h // 2, right_x, sought_y + box_h // 2)
    svg += arrow(left_x + box_w // 2, sought_y + box_h, left_x + box_w // 2, sought_y + box_h + 20)

    svg += section_label(eligibility_y, 100, "Eligibility")
    svg += box(
        left_x,
        eligibility_y,
        box_w,
        box_h,
        f"Reports assessed for eligibility (n = {n_assessed})",
    )
    svg += box(
        right_x,
        eligibility_y,
        excl_w,
        box_h,
        f"Reports excluded (n = {n_excl_elig})",
        excluded=True,
    )
    svg += arrow(
        left_x + box_w,
        eligibility_y + box_h // 2,
        right_x,
        eligibility_y + box_h // 2,
    )
    svg += arrow(
        left_x + box_w // 2,
        eligibility_y + box_h,
        left_x + box_w // 2,
        eligibility_y + box_h + 20,
    )

    svg += section_label(included_y, 70, "Included")
    svg += box(
        left_x,
        included_y,
        box_w,
        55,
        f"Studies included in review (n = {n_included})",
    )

    svg += "</svg>"
    return svg
