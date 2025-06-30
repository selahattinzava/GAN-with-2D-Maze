function rgbImage = drawSquareSafe(rgbImage, occupied, x, y, color)
    for dx = -3:3
        for dy = -3:3
            xx = x + dx;
            yy = y + dy;
            if xx >= 1 && xx <= size(rgbImage,2) && yy >= 1 && yy <= size(rgbImage,1)
                if ~occupied(yy, xx)
                    rgbImage(yy, xx, :) = reshape(color, 1, 1, 3);
                end
            end
        end
    end
end
