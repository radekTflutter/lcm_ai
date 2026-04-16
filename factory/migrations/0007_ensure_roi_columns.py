# Repair: ensure roi_x1, roi_y1, roi_x2, roi_y2 exist (in case 0005 was skipped)

from django.db import migrations, connection


def ensure_roi_columns(apps, schema_editor):
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(factory_camera)")
        columns = [row[1] for row in cursor.fetchall()]
    for col in ['roi_x1', 'roi_y1', 'roi_x2', 'roi_y2']:
        if col not in columns:
            with connection.cursor() as cursor:
                cursor.execute(f"ALTER TABLE factory_camera ADD COLUMN {col} REAL NULL")


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("factory", "0006_add_roi_polygon"),
    ]

    operations = [
        migrations.RunPython(ensure_roi_columns, noop),
    ]
