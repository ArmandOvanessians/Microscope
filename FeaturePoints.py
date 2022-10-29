import json
import sys

class FeaturePoints:
    def __init__(self, filename):
        # load metadata from specified file
        self.load_data(filename)

    def load_data(self, filename: str):
        try:
            # read feature points dictionary from file
            with open(filename) as f:
                json_data = json.load(f)
            self.__dict__ = json_data
        except FileNotFoundError:
            print(f'File not found: {filename}')

    def __str__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())


def write_dict(points, fn):
    try:
        # convert to JSON string and write to file
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump(points, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        print(f'File write error: {fn}')


if __name__ == '__main__':

    for arg in sys.argv[1:]:
        print(arg)

        if '.csv' in arg:
            # command line arg must be QuPath annotations with expected format:
            # CT001-002B_HD.jpg,PathAnnotationObject,,Image,Ellipse,3036,4243.5,9.425,11.05
            with open(arg) as fp:
                lines = fp.readlines()
                fn = ''
                points = {}

                for line in lines[1:]:      # skip header line
                    values = line.split(',')

                    if fn != values[0]:
                        if len(points) > 0:
                            write_dict(points, fn.replace('jpg', 'json'))
                        points.clear()
                        fn = values[0]

                    points['p' + str(len(points))] = (float(values[5]), float(values[6]))

                if len(points) > 0:
                        write_dict(points, fn.replace('jpg', 'json'))

        elif '.json' in arg:
            fp = FeaturePoints(arg)
            print(fp)

