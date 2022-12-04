import random
import numpy as np
import pygame

from sklearn.svm import SVC

WIDTH, HEIGHT = 600, 400
RADIUS = 50


def generate_data(
        elements_number: int, classes_number: int, radius: int = None, centers=None
):
    if not centers:
        centers = [[None, None], [None, None]]
    if not radius:
        radius = RADIUS
    data = []

    for class_number in range(classes_number):
        center_x = centers[class_number][0]
        center_y = centers[class_number][1]
        if not center_x:
            center_x = random.randint(radius, WIDTH - radius)
        if not center_y:
            center_y = random.randint(radius, HEIGHT - radius)

        for _row_number in range(elements_number):
            data.append([[random.gauss(center_x, radius / 2),
                        random.gauss(center_y, radius / 2)], class_number])
    return data


def get_line_points(support_vector_classification: SVC):
    w = support_vector_classification.coef_[0]
    a = w[0] / w[1]
    xx = np.array([0, WIDTH])
    yy = a * xx - (support_vector_classification.intercept_[0]) / w[1]
    return [xx[0], yy[0]], [xx[-1], yy[-1]]


def draw():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    play = True
    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = list(event.pos)
                cls = support_vector_classification.predict([pos])[0]
                points.append([pos, cls])

        for point in points:
            pygame.draw.circle(screen, colors[point[1]], point[0], 3)

        pygame.draw.line(screen, 'white', p1, p2, 2)

        pygame.display.update()


if __name__ == '__main__':
    # Random case:
    points = generate_data(10, 2)

    # 90°:
    # points = generate_data(10, 2, 15, [[300, 50], [300, 350]])

    # 180°:
    # points = generate_data(10, 2, 15, [[51, 200], [198.5, 200]])

    # 45°:
    # points = generate_data(10, 2, 15, [[150, 50], [450, 350]])

    # 135°:
    # points = generate_data(10, 2, 10, [ [150, 350], [450, 50] ])

    colors = {0: 'white', 1: 'red'}
    x_coords = np.array(list(map(lambda p: p[0], points)))
    y_coords = np.array(list(map(lambda p: p[1], points)))
    support_vector_classification = SVC(kernel='linear')
    support_vector_classification.fit(x_coords, y_coords)
    p1, p2 = get_line_points(support_vector_classification)
    draw()
