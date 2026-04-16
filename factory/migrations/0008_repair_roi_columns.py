# Repair: add all ROI columns if missing (roi_x1, roi_y1, roi_x2, roi_y2, roi_polygon)
# Safe to run: only adds columns that don't exist

from django.db import migrations, connection


def repair_roi_columns(apps, schema_editor):
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(factory_camera)")
        columns = [row[1] for row in cursor.fetchall()]
    with connection.cursor() as cursor:
        for col in ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]:
            if col not in columns:
                cursor.execute(f"ALTER TABLE factory_camera ADD COLUMN {col} REAL NULL")
        if "roi_polygon" not in columns:
            cursor.execute("ALTER TABLE factory_camera ADD COLUMN roi_polygon TEXT NULL")


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("factory", "0007_ensure_roi_columns"),
    ]

    operations = [
        migrations.RunPython(repair_roi_columns, noop),
    ]
